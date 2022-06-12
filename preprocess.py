# python preprocess.py -i data/train_both_original_no_cands.txt -o data/train_convo.pickle

from tqdm import tqdm
import os, pickle
import argparse

def process_rawdata(filename):

    raw_data = open(filename).read().strip().split('\n')
    data, count = {}, 0
    curr_convo, curr_ps, curr_pt = [], [], []
    indices = []    

    person_a = 'your persona'
    person_b = "partner's persona"

    with tqdm(total = len(raw_data)) as pbar:
        turn_count, ctx_count = 1,0 #init cycle
        for idx, line in enumerate(raw_data):
            if person_a in line[0:20]:
                if (turn_count != 0) and (len(curr_ps)>1 and len(curr_pt)>1 and len(curr_convo)>1):
                    if idx > 1:
                        if curr_convo[0] == '__SILENCE__' :
                            p1 = curr_ps; p2 = curr_pt; curr_convo = curr_convo[1:]
                        else:
                            p1 = curr_pt; p2 = curr_ps

                        data[count] = { 'convo': curr_convo,
                                        'p_src': p1,
                                        'p_trg': p2}
                        count+=1
                    curr_convo, curr_ps, curr_pt = [], [], []
                    turn_count=0

                words = line.split()
                turn_id, words = int(words[0]), ' '.join(words[3:])
                curr_ps.append(words)

                ctx_count +=1
                assert ctx_count == turn_id
                
            elif person_b in line[0:20]:
                if (turn_count != 0) and (len(curr_ps)>1 and len(curr_pt)>1 and len(curr_convo)>1):
                    if idx > 1:
                        if curr_convo[0] == '__SILENCE__' :
                            p1 = curr_ps; p2 = curr_pt; curr_convo = curr_convo[1:]
                        else:
                            p1 = curr_pt; p2 = curr_ps
                        data[count] = { 'convo': curr_convo[0],
                                        'p_src': p1, 
                                        'p_trg': p2}
                        count+=1
                    curr_convo, curr_ps, curr_pt = [], [], []
                    turn_count=0
                words = line.split()
                turn_id, words = int(words[0]), ' '.join(words[3:])
                curr_pt.append(words)

                ctx_count +=1
                assert ctx_count == turn_id

                
            else:
                if ctx_count !=0:
                    turn_count = ctx_count *1 
                    ctx_count =0
                    indices.append(idx)
                        
                src_line, trg_line = line.split('\t')
                src_words = src_line.split()
                src_idx, src_line = src_words[0], ' '.join(src_words[1:])

                curr_convo.append(src_line) 
                curr_convo.append(trg_line)#turn)
                
                turn_count +=1
                assert turn_count == int(src_idx)
                
            pbar.update(1)

    return data, count

def print_convo(conv_ex):

    print("Person A persona: ")
    for pt in conv_ex['p_src']:
        print(pt)
    print('')

    print("Person B persona: ")
    for pt in conv_ex['p_trg']:
        print(pt)
    print('')

    turn = 0
    print("Conversation b/w them: ")
    for ut in conv_ex['convo']:
        if turn == 0:
            print('Person A: ', ut)
        else:
            print('Person B: ', ut)
        turn += 1
        turn %= 2 

parser = argparse.ArgumentParser()                                               

parser.add_argument("--input", "-i", type=str, required=True)
parser.add_argument("--output", "-o", type=str, required=True)
args = parser.parse_args()

convos, count = process_rawdata(args.input)
assert(len(convos) == count)

print_convo(convos[10])

with open(args.output, 'wb') as handle:
    pickle.dump(convos, handle, protocol=pickle.HIGHEST_PROTOCOL)
