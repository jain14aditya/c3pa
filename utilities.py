import torch

flatten = lambda l: [item for sublist in l for item in sublist]

# torch.cuda.is_available = lambda : False

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def tokenize_conv(conv, tokenizer):
    '''tokenize personas and the complete chat'''
    persona_a = tokenizer.encode(''.join(['<|p1|>'] + conv['p_src'] + ['<|sep|>'] + ['<|start|>']))
    persona_b = tokenizer.encode(''.join(['<|p2|>'] + conv['p_trg'] + ['<|sep|>'] + ['<|start|>']))

    chat = []
    for i in range(len(conv['convo']) - 1):
        chat += [tokenizer.encode(conv['convo'][i] + tokenizer.eos_token)]

    return persona_a, persona_b, chat

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def generate_next(model, tokenizer, bot_input_ids, do_sample=True, top_k=10, top_p=.92, max_length=1000):
    msg_tensor = model.generate(bot_input_ids, do_sample=True,
                                              top_k=top_k, top_p=top_p, 
                                              max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    msg = to_data(msg_tensor.detach()[0])[bot_input_ids.shape[-1]:]
    return msg_tensor[0][bot_input_ids.shape[-1]:], msg

def load_query_labels(persona_a, persona_b, chat):
    '''generates multiple queries and the corresponding actual human responses for a chat'''
    queries = []
    labels = []
    query_len = []

    for bot in range(2):
        bot_persona = persona_a if bot%2 == 0 else persona_b
        for i in range(len(chat)):
            if i%2 == bot%2:
                bot_input_ids = [bot_persona + flatten(chat[:i])]
                queries.append(bot_input_ids)
                labels.append(chat[i])
                query_len.append(i)

    assert(len(chat) == len(queries))
    return queries, labels, query_len


def tokenized_queries(train_convs, tokenizer, num_convs = float('inf')):

    train_queries = []
    train_labels = []
    train_query_len = []
    train_conv_ids = []

    convs_processed = 0
    for conv_id in train_convs:
        persona_a, persona_b, chat = tokenize_conv(train_convs[conv_id], tokenizer)
        queries, labels, query_len = load_query_labels(persona_a, persona_b, chat)
        train_queries += queries
        train_labels += labels
        train_query_len += query_len
        train_conv_ids += [conv_id]*len(queries)    
        convs_processed += 1
        if convs_processed > num_convs:
            break

    assert(len(train_queries) == len(train_labels) == len(train_conv_ids))
    # print(len(train_queries))

    return train_queries, train_labels, train_query_len, train_conv_ids