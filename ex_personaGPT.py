import os, pickle
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.sentiment import SentimentIntensityAnalyzer

processed_data = 'data/train_convo.pickle'
model_name = "af1tang/personaGPT"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

if torch.cuda.is_available():
    model = model.cuda()

with open(processed_data, 'rb') as handle:
    convos = pickle.load(handle)

flatten = lambda l: [item for sublist in l for item in sublist]

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_length=1000, pad_token=tokenizer.eos_token_id):
    full_msg = model.generate(bot_input_ids, do_sample=True,
                                              top_k=top_k, top_p=top_p, 
                                              max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
    return msg

def tokenize_conv(conv):
    
    persona_a = tokenizer.encode(''.join([tokenizer.pad_token]*10 + ['<|p1|>'] + conv['p_src'] + ['<|sep|>'] + ['<|start|>']))
    persona_b = tokenizer.encode(''.join([tokenizer.pad_token]*10 + ['<|p2|>'] + conv['p_trg'] + ['<|sep|>'] + ['<|start|>']))

    chat = []
    for i in range(len(conv['convo']) - 1):
        chat += [tokenizer.encode(conv['convo'][i] + tokenizer.eos_token)]

    return persona_a, persona_b, chat

sia = SentimentIntensityAnalyzer()

conv = convos[0]

print('------------- Persona A ------------')
for pd in conv['p_src']:
    print(pd)
print('------------- Persona B ------------')
for pd in conv['p_trg']:
    print(pd)
print('------------- Chat ------------')

persona_a, persona_b, chat = tokenize_conv(conv)

for bot in range(2):
    bot_persona = persona_a if bot%2 == 0 else persona_b
    for i in range(len(chat)):
        if i%2 == bot%2:
            if i == 0:
                bot_input_ids = to_var([bot_persona]).long()
            else:
                bot_input_ids = to_var([bot_persona + flatten(chat[:i])]).long()
            msg = generate_next(bot_input_ids)
            decoded_msg = tokenizer.decode(msg, skip_special_tokens=True)
            label = tokenizer.decode(chat[i], skip_special_tokens=True)

            print("Bot: {} | Sentimenet : {}".format(decoded_msg, sia.polarity_scores(decoded_msg)['pos']))
            print("Actual: {} | Sentimenet : {}".format(label, sia.polarity_scores(label)['pos']))
            continue
        print('Human ' , i%2, ': ', tokenizer.decode(chat[i], skip_special_tokens=True))
    print('**************************************************************************')