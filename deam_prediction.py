import argparse
from faulthandler import disable
import os
import random
import numpy as np
import torch
# from scipy.stats import spearmanr
import json

from transformers import (
    Trainer,
    TrainingArguments,
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast)
# from sklearn.metrics import f1_score, accuracy_score

from timeit import default_timer as timer

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
# device='cpu'
# print(torch.cuda.current_device())
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     # torch.cuda.set_device(0)
#     torch.cuda.manual_seed_all(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DeamPredict:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_data_path', default='./data/topical_persona/train_amr_manamr_cont_coref_pirel_eng.txt', required=False, help="path of train conversations")
        parser.add_argument('--valid_data_path', default='./data/topical_persona/valid_amr_manamr_cont_coref_pirel_eng.txt', required=False, help="path of valid conversations")
        parser.add_argument('--model_path', default='coh_models/', required=False, help="path of trained model")
        parser.add_argument('--max_length', default=512, type=int, required=False, help="maximum length of input conversations")
        parser.add_argument('--num_labels', default=2, type=int, required=False, help="number of labels for classifying conversations")
        parser.add_argument('--num_epochs', default=3, type=int, required=False, help="number of training epochs")
        parser.add_argument('--train_batch_size', default=8, type=int, required=False, help="batch size for training")
        parser.add_argument('--valid_batch_size', default=8, type=int, required=False, help="batch size for evaluating")
        parser.add_argument('--warmup_steps', default=500, type=int, required=False, help="number of warmup steps for lr scheduler")
        parser.add_argument('--warmup_decay', default=0.01, required=False, help="weight decay value")
        parser.add_argument('--model_name', default='roberta-large', required=False, help="name of the model")
        parser.add_argument('--model_type', default='roberta', required=False, help="type of the model")
        parser.add_argument('--logging_steps', default=100, type=int, required=False, help="logging model weights")
        parser.add_argument('--save_steps', default=200, type=int, required=False, help="save model weights")
        parser.add_argument('--learning_rate', default=0.0001, type=float, required=False, help="learning rate")
        parser.add_argument('--mode', default='predict',  required=False, help="mode (train/valid/predict)")
        parser.add_argument("--gpu", type=int, default=1, help="increase output verbosity")
        args=parser.parse_args()
        self.args = args
        MODEL_CLASSES = {
        "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast),
        }
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        if args.mode=='train':
            self.tokenizer = tokenizer_class.from_pretrained(args.model_name, do_lower_case=True)
            self.model = model_class.from_pretrained(args.model_name, num_labels=args.num_labels).to(device)
        else:
            # print(args.model_name)
            self.tokenizer = tokenizer_class.from_pretrained(args.model_path)
            self.model = model_class.from_pretrained(args.model_path, num_labels=args.num_labels).to(device)
        # print(self.model)
        training_args = TrainingArguments(output_dir=args.model_path+'/'+args.model_type, log_level='error', disable_tqdm=True)
        self.trainer = Trainer(model=self.model, args=training_args, tokenizer=self.tokenizer)
        # print(self.predict("Hi</UTT>yo</UTT>Hello</UTT>Bye</UTT>"))

    def get_convs_from_input(self, input_convs):
        convs=[]
        orig_convs=[]
        line=input_convs.split('\n')[0]
        if not line:
            print('error line is an empty conversation')
        else:
            conv=line.split('\n')[0]
            parts=line.split('</UTT>')
            conv=' '.join(parts[:-1])
            convs.append(conv)
            orig_convs.append('</UTT>'.join(parts[:-1]))
        return convs, orig_convs
    
    def predict(self, convs):
        test_convs, test_orig_convs = self.get_convs_from_input(convs)
        # print(test_convs)
        encodings = self.tokenizer(test_convs, truncation=True, padding=True, max_length=512)
        # print(encodings)
        test_dataset=TestDataset(encodings)
        
        output=self.trainer.predict(test_dataset)
        prob=torch.nn.Softmax(dim=1)
        scores=prob(torch.tensor(output.predictions))
        return scores[0][1].item()

    def coherence_score(self, convs):
        input_conv = '</UTT>'.join(convs)
        input_conv += '</UTT>'
        # print(input_conv)
        return self.predict(input_conv)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        '''
        Param:
            encodings: encodings of inputs
            labels: labels of inputs
        '''
        self.encodings = encodings
        print("self.encodings.keys() in Dataset", self.encodings.keys())
        self.labels = labels

    def __getitem__(self, idx):
        '''get the specifeid item's encodings and label 
        Param:
            idx: index of an input
        '''
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        '''number of data
        '''
        if not self.labels:
            return 0
        return len(self.labels)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        '''
        Param:
            encodings: encodings of inputs
        '''
        self.encodings = encodings

    def __getitem__(self, idx):
        '''get the specifeid item's encodings and label 
        Param:
            idx: index of an input
        '''
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item

    def __len__(self):
        '''number of data
        '''
        return len(self.encodings.input_ids)


def load_data(data_path):
    '''load data
        Param:
            data_path: path of the input data
    '''
    fr = open(data_path, 'r')
    lines=fr.readlines()
    convs=[]
    orig_convs=[]
    labels=[]
    for ind, line in enumerate(lines):
        line=line.split('\n')[0]
        if not line:
            print('error line {} is an empty conversation'.format(ind))
        else:
            parts=line.split('</UTT>')
            label=round(float(parts[-1]))
            conv=' '.join(parts[:-1])
            convs.append(conv)
            orig_convs.append('</UTT>'.join(parts[:-1]))
            labels.append(label)
    return convs, orig_convs, labels

def load_test_data(data_path):
    '''load test data
        Param:
            data_path: path of the input test data
    '''
    fr = open(data_path, 'r')
    lines=fr.readlines()
    convs=[]
    orig_convs=[]
    for ind, line in enumerate(lines):
        line=line.split('\n')[0]
        if not line:
            print('error line {} is an empty conversation'.format(ind))
        else:
            conv=line.split('\n')[0]
            parts=line.split('</UTT>')
            conv=' '.join(parts[:-1])
            convs.append(conv)
            orig_convs.append('</UTT>'.join(parts[:-1]))
    return convs, orig_convs

def load_and_cache_examples(args, data, labels=None, type_data="train", additional_filename_postfix=""):
    '''load  and cache input data
        Param:
            data: input data
            labels: label of data
            type_data: whether it is train/valid/test
            additional_filename_postfix: which test set
    '''
    if not os.path.exists(args.model_path):
        os.mkdir(os.path.join('./',args.model_path))
    cached_features_file = os.path.join(args.model_path,"cached_{}_{}_{}{}".format(type_data, args.model_type, args.max_length, additional_filename_postfix))
    if os.path.exists(cached_features_file):
        encodings = torch.load(cached_features_file)
    else:
        encodings = tokenizer(data, truncation=True, padding=True, max_length=args.max_length)
        torch.save(encodings, cached_features_file)
    if labels:
        dataset=Dataset(encodings, labels)
    else:
        dataset=TestDataset(encodings)
    print("len(dataset) is {} in load_and_cache_examples with type_data {}".format(len(dataset), type_data))
    return dataset


def get_metrics(output):
    '''return accuracy and f1 scores
        Param:
            output: ground-truth and predicted scores
    '''
    labels = output.label_ids
    preds = output.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1=f1_score(labels,preds)
    return {'accuracy': acc, 'f1': f1}

# if __name__ == "__main__":
#     start = timer()
#     deamPredict = DeamPredict()
#     end = timer()
#     print("Time elapsed = ", end - start)
    
#     start = timer()
#     print(deamPredict.predict("Hi!</UTT>Yo</UTT>How are You</UTT>I am fine</UTT>How is life going</UTT>What are you doing</UTT>What is the meeting about?</UTT>I cannot tell you</UTT>What can you tell me?</UTT></UTT>I am god</UTT>I just finished my homework</UTT>"))
#     end = timer()
#     print("Time elapsed = ", end - start)

#     start = timer()
#     print(deamPredict.predict("Hi!</UTT>Yo</UTT>How are You</UTT>I am fine</UTT>How is life going</UTT>Busy with meetings</UTT>What is the meeting about?</UTT>I cannot tell you</UTT>What can you tell me?</UTT>Nothing much I am in love</UTT>"))
#     end = timer()
#     print("Time elapsed = ", end - start)

#     start = timer()
#     print(deamPredict.predict("hi, how are you doing? i'm getting ready to do some cheetah chasing to stay in shape.</UTT>you must be very fast. hunting is one of my favorite hobbies.</UTT>i am! for my hobby i like to do canning or some whittling.</UTT>i also remodel homes when i am not out bow hunting.</UTT>that's neat. when i was in high school i placed 6th in 100m dash!</UTT>that's awesome. do you have a favorite season or time of year?</UTT>i do not. but i do have a favorite meat since that is all i eat exclusively.</UTT>what is your favorite meat to eat?</UTT>i would have to say its prime rib. do you have any favorite foods?</UTT>i like chicken or macaroni and cheese.</UTT>do you have anything planned for today? i think i am going to do some canning.</UTT>i am going to watch football. what are you canning?</UTT>i think i will can some jam. do you also play footfall for fun.</UTT>"))
#     end = timer()
#     print("Time elapsed = ", end - start)