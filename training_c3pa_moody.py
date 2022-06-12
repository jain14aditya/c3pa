from cProfile import label
import os, pickle, time
import string
from statistics import mean
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import random

from utilities import *
import deam_prediction
import wandb

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch

parser = argparse.ArgumentParser()
# parser.add_argument("w1", help="sentiment weight", type=float)
# parser.add_argument("w2", help="coherence weight", type=float)
# parser.add_argument("name", help="model display name", type=str)
# parser.add_argument("device_num", help="gpu device to run your experiments on", type=str)
# args = parser.parse_args()

w1 = 1
w2 = 0
name = 'moody_c3pa_v2'

# #### Debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch.cuda.is_available = lambda : False

# Config of the training
config = {
    "name": "run-"+name,
    "model_name": "af1tang/personaGPT",
    "save_path": "c3pa_models/"+name+'/',
    "device": "cuda",
    "sentiment_lambda": w1,
    "coherence_lambda": w2,
    "train_data": 'data/train_convo.pickle',
    "steps": 20000,
    "batch_size": 1,
    "forward_batch_size": 1,
    "ppo_epochs": 2,
    "tracking_count": 1000,
    "save_after": 20000,
    "txt_in_min_len": 2,
    "txt_in_max_len": 8,
    "txt_out_min_len": 4,
    "txt_out_max_len": 16,
    "lr": 1.41e-6,
    "init_kl_coef":0.4,
    "adap_kl_ctrl":True,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
}

# Initiliazing the device to be trained on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

print('\n###################################### Training config ##################################################\n')
print('Training c3pa model stored at : {} | with weights : {} and {} on {} \n'.format(name, w1, w2, device))

# Loading model, tokenizer, and the ppo_trainer
print('\n###################################### Step 1: Loading all the models ##################################################\n')
gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
gpt2_model.to(device)
gpt2_model_ref.to(device)
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, tokenizer, **config)
# Reward Scorers
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    # "batch_size": config["forward_batch_size"]
}
sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)
# ### example on how to use the sentiment scorer
# text = 'this movie was really bad!!'
# sen_ex1 = sentiment_pipe(text, **sent_kwargs)
# print(sen_ex1)

# text = 'this movie was really good!!'
# sen_ex2 = sentiment_pipe(text, **sent_kwargs)
# print(sen_ex2)

# Initializing the Wandb logger
wandb.init(name=config["name"], project='c3pa', config=config, )
wandb.watch(gpt2_model, log='all')

# Loading Data
print('\n###################################### Step 2: Processing the training data ##################################################\n')
with open(config['train_data'], 'rb') as handle:
    train_convs = pickle.load(handle)
# process them as queries, human responses and the corresponding convs
train_queries, train_labels, train_query_len, train_conv_ids = tokenized_queries(train_convs, tokenizer)
assert(len(train_queries) == len(train_labels) == len(train_query_len) == len(train_conv_ids))

# #### Randomize the order of the queries
# training_set = list(zip(train_queries, train_labels, train_query_len, train_conv_ids))
# random.shuffle(training_set)
# train_queries, train_labels, train_query_len, train_conv_ids = zip(*training_set)

print("Processed {} conversations resulting in {} queries.".format(len(train_convs), len(train_queries)))

print('\n###################################### Step 3: PPO Training ##################################################\n')

checkpoint_num = 0
avg_reward = []

for epoch in range(config['ppo_epochs']):
    print('\n ---------------------------------------- Epoch {} ------------------------------------------------------------- \n'.format(epoch))
    lastn_rewards = []
    for i in range(len(train_queries)):
        # print('\n ********************* Processing query num : {} ***************************** \n'.format(i))
        logs, timing = dict(), dict()
        t0 = time.time()
        
        query_tensors = to_var(train_queries[i]).long()
        decoded_query = tokenizer.decode(to_data(query_tensors.detach()[0]), skip_special_tokens=False)
        human_response = train_convs[train_conv_ids[i]]['convo'][train_query_len[i]]
        # print('\n Query -----> {} \n'.format(decoded_query))
        # print('\n Human response -----> {} \n'.format(human_response))

        #### Get response from gpt2
        # print('\n sub-step 1 : Getting response from the model \n')
        t = time.time()
        response_tensor, response = generate_next(gpt2_model, tokenizer, query_tensors)
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)
        timing['time/get_response'] = time.time()-t
        # print('\n Generated Response -----> {} \n'.format(decoded_response))

        #### Compute sentiment score
        # print('\n subStep 2 : Computing the reward \n')
        t = time.time()
        hum_pipe_outputs = sentiment_pipe(human_response, **sent_kwargs)
        gen_pipe_outputs = sentiment_pipe(decoded_response, **sent_kwargs)
        reward_score = 0
        if hum_pipe_outputs[0][0]["score"] > 0:
            reward_score = gen_pipe_outputs[0][0]["score"]
        elif hum_pipe_outputs[0][1]["score"] > 0:
            reward_score = gen_pipe_outputs[0][1]["score"]
        reward = torch.tensor([reward_score]).to(device)
        timing['time/get_rewards'] = time.time()-t
        # print('\n Reward -----> {} \n'.format(reward[0].item()))

        # #### Run PPO step 
        # print('\n subStep 2 : Running PPO')
        t = time.time()
        stats = ppo_trainer.step([query_tensors[0]], [response_tensor], reward)
        timing['time/optimization'] = time.time()-t        
        lastn_rewards.append(reward[0].item())

        # #### Log timing and ppo stats at every step
        logs.update(timing)
        logs.update(stats)
        logs['env/query'] = decoded_query
        logs['env/response'] = decoded_response
        logs['env/reward'] = reward[0].item()
        del query_tensors, response_tensor, reward

        #### Tracking the model
        if i>0 and i%config['tracking_count'] == 0:
            print("\n -------------------------- Model stats after {} training iterations ----------------------------------------- \n".format(i))
            print('\n Averaged reward over last {} runs is {} \n'.format(config['tracking_count'], mean(lastn_rewards[-config['tracking_count']:])))
            logs['rewards/overall'] = mean(lastn_rewards[-config['tracking_count']:])
            if i%config['save_after'] == 0:
                print("\n -------------------------- Saving model as checkpoint_{} after {} training iterations ----------------------------------------- \n".format(checkpoint_num, i))
                gpt2_model.save_pretrained(config['save_path']+'checkpoint_'+str(checkpoint_num))
                checkpoint_num += 1
            
        wandb.log(logs)

print('\n ###################################### Step 4: Saving the Model at {} ################################################## \n'.format(config['save_path']+ 'checkpoint_' + str(checkpoint_num)))
gpt2_model.save_pretrained(config['save_path']+'checkpoint_'+str(checkpoint_num))