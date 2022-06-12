from cProfile import label
import os, pickle, time
import string
import numpy as np
import pandas as pd
import random

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch

gpt2_model = GPT2HeadWithValueModel.from_pretrained("c3pa_models/hybrid_exp1_v1")
# gpt2_model = GPT2HeadWithValueModel.from_pretrained("c3pa_models/coh_c3pa/checkpoint_0")
# gpt2_model = GPT2HeadWithValueModel.from_pretrained("c3pa_models/pos_c3pa_v2/checkpoint_2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")

if torch.cuda.is_available():
    gpt2_model = gpt2_model.cuda()

flatten = lambda l: [item for sublist in l for item in sublist]

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = gpt2_tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()

def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_length=1000, pad_token=gpt2_tokenizer.eos_token_id):
    full_msg = gpt2_model.generate(bot_input_ids, do_sample=True,
                                              top_k=top_k, top_p=top_p, 
                                              max_length=max_length, pad_token_id=gpt2_tokenizer.eos_token_id)
    msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
    return msg

# get personality facts for conversation
personas = []
for i in range(3):
    response = input(">> Fact %d: "%(i+1))+ gpt2_tokenizer.eos_token
    personas.append(response)
personas = gpt2_tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))

# converse for 8 turns
dialog_hx = []
for step in range(8):
    # encode the user input
    user_inp = gpt2_tokenizer.encode(input(">> User: ") + gpt2_tokenizer.eos_token)
    # append to the chat history
    dialog_hx.append(user_inp)
        
    # generated a response while limiting the total chat history to 1000 tokens, 
    bot_input_ids = to_var([personas + flatten(dialog_hx)]).long()
    msg = generate_next(bot_input_ids)
    dialog_hx.append(msg)
    print("Bot: {}".format(gpt2_tokenizer.decode(msg, skip_special_tokens=True)))