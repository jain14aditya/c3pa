#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:10:21 2020

@author: af1tang
"""
import torch, os, pickle, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from itertools import groupby

from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from transformers import pipeline
import deam_prediction
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


load_dotenv(verbose=True)

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

save_path = os.getenv("save_path")
tokenizer_path = os.path.join(save_path, 'checkpoint/tokenizer/')
model_path = os.path.join(save_path, 'checkpoint/model/')
data_path = os.getenv("data_path")
# learning
lr = os.getenv("learn_rate")
gradient_accumulation_steps = os.getenv("gradient_accumulation_steps")
bs = os.getenv("batch_size")
epochs = os.getenv("epochs")
weight_decay = os.getenv("weight_decay")
logging_steps = os.getenv("logging_steps")
save_steps = os.getenv("save_steps")

def create_dir(directory):
    """create directory if not exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# initialize save folder
create_dir(save_path)

class Configs():
    def __init__(self):
        # saving and loading paths
        self.model_path = os.path.join(save_path,'checkpoint/model/')
        self.raw_data_path = os.path.join(save_path, 'train_data')
        self.val_data_path = os.path.join(save_path, 'valid_data')
        self.active_data_path = os.path.join(data_path, 'active_data')
        self.output_dir = os.path.join(save_path, 'checkpoint/model/')
        self.model_name_or_path = os.path.join(save_path,'checkpoint/model/')
        self.plot_path = os.path.join(save_path,'samples/')
        self.download_name = 'microsoft/DialoGPT-medium'
        self.i2p_path = os.path.join(save_path, 'i2p')
        # eval
        self.do_eval = True
        self.evaluate_during_training = False
        # batching
        self.batch_size = int(bs)
        self.eval_batch_size = 1
        # optimization
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.eps = float(1e-8)
        self.max_grad_norm = 1.0
        self.num_train_epochs = int(epochs)
        self.max_steps = -1
        self.warmup_steps = 0
        # logging
        self.logging_steps = int(logging_steps)
        self.save_steps = int(save_steps)
        # fp16
        self.use_token_ids = False
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        # sampling params
        self.top_k = 20
        self.top_p = .92
        
opts = Configs()

model = None
tokenizer = None
p1_tok, p2_tok, start_tok = None, None, None
act_tok = None

flatten = lambda l: [item for sublist in l for item in sublist]
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
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

def process_conv(row, tokenizer, eos = True, make_flat=True):
    if eos:
        conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row])
    else: conv = list([tokenizer.encode(x) for x in row])
    if make_flat: conv = flatten(conv)
    return conv

def split_by_index(seq, sep):
    result = []
    for el in seq:
        result.append(el)
        if el == sep:
            yield result
            result = []
            
# def filter_turn_indices(x):
#     filtered = [[t[1] for t in list(g)] for k,g in groupby(list(enumerate(x)), lambda x: x[1]==tokenizer.eos_token_id) if not k]
#     return filtered

def filter_turn_indices(x):
    filtered = [[t[1] for t in list(g)] for k,g in groupby(list(enumerate(x)), lambda x: x[1]==tokenizer.eos_token_id or x[1] == p1_tok or x[1] == p2_tok or x[1] == start_tok or x[1] == tokenizer.sep_token) if not k]
    return filtered

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()

### plotting ###    
def plot_losses(stats, title='loss'):
    create_dir(opts.plot_path)
    x = list(sorted(stats.keys()))
    loss = [stats[i][title] for i in x]
    plt.plot(x, loss, label= title)
    plt.legend()
    plt.title("%s" %title)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.plot_path,'%s.png'%title))
    plt.close()

## model saving ##
def checkpoint(model, tokenizer, optimizer, scheduler, stats, title=""):
    create_dir(opts.output_dir)
    model.save_pretrained(opts.output_dir)
    tokenizer.save_pretrained(opts.output_dir)
    torch.save(opts, os.path.join(opts.output_dir, title+"training_opts.bin"))
    torch.save(optimizer.state_dict(), os.path.join(opts.output_dir, title+'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(opts.output_dir, title+'scheduler.pt'))
    with open(os.path.join(opts.output_dir, title+'stats.pkl'), 'wb') as f: pickle.dump(stats,f)
    
## Training Pipeline ##
def fit_on_batch(batch):
    xx,yy = batch
    try:
        xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
    except:
        xx, yy = to_var(xx), to_var(yy)
    ## forward on new data batch
    _outp = model(xx)
    past = _outp.past_key_values
    outp = model(yy, past_key_values=past, labels=yy)
    
    # backward
    loss = outp[0]; del outp
    if opts.gradient_accumulation_steps > 1:
        loss = loss / opts.gradient_accumulation_steps
    loss.backward()
    return loss

print('\n###################################### Step 1: Loading all the models ##################################################\n')
pipe_device = 0 if torch.cuda.is_available() else -1
sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    # "batch_size": config["forward_batch_size"]
}
coherence_scorer = deam_prediction.DeamPredict()
t = coherence_scorer.coherence_score("Hello world")
print("Dummy score = ", t)
sia = SentimentIntensityAnalyzer()
print('\n###################################### Step 1 Done : Loading all the models ##################################################\n')

def evaluate_loop(data):
    dataloader = DataLoader(data, batch_size=1, shuffle=True); del data
    data_iter = iter(dataloader)
    print("total = ", len(dataloader))
    with torch.no_grad():
        eval_stats, total_steps, val_loss, val_f1_score = {}, 0, 0.0, 0.0
        pos_sent_score_ytrue, pos_sent_score_ypred = 0.0, 0.0
        coherence_score_ytrue, coherence_score_ypred = 0.0, 0.0 
        model.eval()
        for i in tqdm(range(len(dataloader))):
            batch = data_iter.next()

            xx,yy = batch
            # print("xx = ", len(xx), " yy = ", len(yy))
            try:
                xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
            except:
                xx, yy = to_var(xx), to_var(yy)
            _outp = model(xx)
            past = _outp.past_key_values
            outp = model(yy, past_key_values=past, labels=yy)
            # print("outp = ", outp)
            loss = outp[0]
            # print("loss = ", loss)
            # xx_np = np.array( filter_turn_indices(to_data(xx[...,1:].contiguous().view(-1)) ))
            xx_np = np.array( filter_turn_indices(to_data(xx.contiguous().view(-1)) ))[-2:]
            # print("xx_np shape = ", xx_np.shape)
            ytrue=np.array( filter_turn_indices(to_data(yy[...,1:].contiguous().view(-1)) ) )
            ypred=np.array( filter_turn_indices(to_data( outp[1][..., :-1, :].contiguous().topk(1)[1].view(-1)) ) ) 
            # print("y_test shape = ", ytrue.shape, "y_pred shape = ", ypred.shape)
            # decoded_xx = [tokenizer.decode(p) for p in xx]
            # decoded_yy = [tokenizer.decode(p) for p in yy]
            # print("decoded_xx shape = ", decoded_xx)
            # print("decoded_yy shape = ", decoded_yy)
            ytrue = ytrue[:1]
            ypred = ypred[:1]
            # print("yture = ", len(ytrue), len(ytrue[0]))
            decoded_xx = [tokenizer.decode(flatten(xx_np))]
            decoded_ytrue = [tokenizer.decode(ytrue[0])]
            decoded_ypred = [tokenizer.decode(ypred[0])]
            # decoded_ytrue = [tokenizer.decode(p) for p in ytrue]
            # decoded_ypred = [tokenizer.decode(p) for p in ypred]
            # print("decoded_xx shape = ", len(decoded_xx))
            # print("decoded_xx shape = ", decoded_xx)
            # print("decoded_ytrue shape = ", decoded_ytrue)
            # print("decoded_ypred shape = ", decoded_ypred)

            ########################################
            # Sentiment
            # pipe_outputs_ytrue = sentiment_pipe(decoded_ytrue, **sent_kwargs)[0][1]['score']
            # pipe_outputs_ypred = sentiment_pipe(decoded_ytrue, **sent_kwargs)[0][1]['score']
            # # print("sentiment_pipe(decoded_ytrue, **sent_kwargs).shape = ", sentiment_pipe(decoded_ytrue, **sent_kwargs))
            # # print("pipe_outputs_ytrue = ", pipe_outputs_ytrue, " pipe_outputs_ypred = ", pipe_outputs_ypred)
            # pos_sent_score_ytrue += pipe_outputs_ytrue
            # pos_sent_score_ypred += pipe_outputs_ypred
            ########################################

            ########################################
            # DEAM
            # reward_coherence_ytrue = coherence_scorer.coherence_score(decoded_ytrue)
            # reward_coherence_ypred = coherence_scorer.coherence_score(decoded_ypred)
            xx_ytest_l = decoded_xx + decoded_ytrue
            xx_ypred_l = decoded_xx + decoded_ypred
            st = ''.join(str(x) for x in xx_ytest_l)
            sp = ''.join(str(x) for x in xx_ypred_l)
            # print("st = ", st)
            # print("sp = ", sp)
            reward_coherence_ytrue = coherence_scorer.coherence_score(st)
            reward_coherence_ypred = coherence_scorer.coherence_score(sp)
            # print("reward_coherence_ytrue = ", reward_coherence_ytrue, " reward_coherence_ypred = ", reward_coherence_ypred)
            coherence_score_ytrue += reward_coherence_ytrue
            coherence_score_ypred += reward_coherence_ypred
            ########################################


            #################################################################
            # Sentiment
            senti_xx_ytrue = sia.polarity_scores(st)['pos']
            senti_xx_ypred = sia.polarity_scores(sp)['pos']
            # pipe_outputs_ytrue = sentiment_pipe(decoded_ytrue, **sent_kwargs)[0][1]['score']
            # pipe_outputs_ypred = sentiment_pipe(decoded_ytrue, **sent_kwargs)[0][1]['score']
            # # print("sentiment_pipe(decoded_ytrue, **sent_kwargs).shape = ", sentiment_pipe(decoded_ytrue, **sent_kwargs))
            # print("senti_xx_ytrue = ", senti_xx_ytrue, " senti_xx_ypred = ", senti_xx_ypred)
            pos_sent_score_ytrue += senti_xx_ytrue
            pos_sent_score_ypred += senti_xx_ypred
            ##################################################################
            
            min_len = min(len(ypred), len(ytrue))
            # print("min_len = ", min_len)
            hits = [set(ypred[i]).intersection(set(ytrue[i])) for i in range(min_len)]
            prec = [len(hits[i])/len(ypred[i]) for i in range(min_len)]
            rec = [len(hits[i])/len(ytrue[i]) for i in range(min_len)]
            f1 = np.mean([2*(prec[i]*rec[i])/(prec[i] + rec[i]+1e-3) for i in range(min_len)])
            val_f1_score += f1
            val_loss += loss.mean().item()
            total_steps +=1 
            # if total_steps%100 ==0: print("... %d out of %d"%(total_steps, len(dataloader)))
            
    val_loss = val_loss / total_steps 
    val_f1_score = val_f1_score / total_steps
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    eval_stats = {'perplexity': perplexity, 'loss': val_loss, 'f1': val_f1_score}
    

    pos_sent_score_ytrue_avg =  pos_sent_score_ytrue / total_steps
    pos_sent_score_ypred_avg =  pos_sent_score_ypred / total_steps
    print("pos_sent_score_ytrue_avg = ", pos_sent_score_ytrue_avg)
    print("pos_sent_score_ypred_avg = ", pos_sent_score_ypred_avg)

    coherence_score_ytrue_avg = coherence_score_ytrue / total_steps
    coherence_score_ypred_avg = coherence_score_ypred / total_steps
    print("coherence_score_ytrue_avg = ", coherence_score_ytrue_avg)
    print("coherence_score_ypred_avg = ", coherence_score_ypred_avg)
    print("Done.")
    return eval_stats

if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", help="increase output verbosity")
    args = parser.parse_args()
    print("eval folder = ", args.eval)

    opts.model_name_or_path = os.path.join(os.getenv("save_path"), args.eval + "/")
    print("Load tokenizer from = ", opts.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(opts.model_path, 
                                        pad_token='<|endoftext|>', cls_token='<|cls|>',
                                        sep_token='<|sep|>')
    print("Load pre-trained model from = ", opts.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(opts.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(opts.model_path)
                                        # pad_token='<|endoftext|>', cls_token='<|cls|>',
                                        # sep_token='<|sep|>')
    # print("Load pre-trained model from = ", opts.model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(opts.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained("af1tang/personaGPT")
    # model = AutoModelForCausalLM.from_pretrained("af1tang/personaGPT")

    tokenizer.add_special_tokens({'additional_special_tokens': ['<|start|>', '<|p1|>', '<|p2|>', '<|act|>']})
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    p1_tok, p2_tok, start_tok = tokenizer.encode('<|p1|>')[0], tokenizer.encode('<|p2|>')[0], tokenizer.encode('<|start|>')[0]
    # new, action token
    act_tok = tokenizer.encode('<|act|>')[0]

    print("="*50)
    print("Evaluating ... ")
    with open(opts.val_data_path, 'rb') as f: eval_data = pickle.load(f)
    eval_stats = evaluate_loop(eval_data)
    print("Done!")
    print()
    print("Perplexity: %.5f" %eval_stats['perplexity'])
    print("F1 Score: %.5f" % eval_stats['f1'])
    print("F1 Score x 100 : %.5f" % (eval_stats['f1'] * 100))
    print("="*50)
