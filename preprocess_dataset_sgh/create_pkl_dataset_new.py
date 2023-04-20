'''
removes question mark 
'''

import matplotlib.pyplot as plt
import os
import json
import pickle
import torch
import numpy as np
import re
#from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.utils import shuffle
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--data-path',type=str,default='',help='path to the directory that stores cleaned_sgh_train,test and val')

args = parser.parse_args()
path = args.data_path


model_type = 'xlm-roberta-base' #albert-base-v1, bert-base-cased, bert-base-uncased, xlm-roberta-base
train_data_path = os.path.join(path,'cleaned_sgh_train.txt')
test_data_path = os.path.join(path,'cleaned_sgh_test.txt')
val_data_path = os.path.join(path,'cleaned_sgh_val.txt')



with open(train_data_path, 'r', encoding='utf-8') as f:
    train_text = f.readlines()
with open(val_data_path, 'r', encoding='utf-8') as f:
    valid_text = f.readlines()
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_text = f.readlines()


datasets = train_text, valid_text, test_text

print([len(ds) for ds in datasets])

def clean_text(text):
    text = text.replace('！', '。')
    text = text.replace('：', '，')
    text = text.replace('——', '，')
    
    #reg = "(?<=[a-zA-Z])-(?=[a-zA-Z]{2,})"
    #r = re.compile(reg, re.DOTALL)
    #text = r.sub(' ', text)
    
    text = re.sub(r'\s—\s', ' ， ', text)
    
#     text = text.replace('-', ',')
    text = text.replace(';', '。')    # replace symbols with the most relevant counterparts
    text = text.replace('、', '，')
    text = text.replace('♫', '')
    text = text.replace('……', '')
    text = text.replace('。”', '')
    text = text.replace('”', '，')
    text = text.replace('“','，')
    text = text.replace(',','，')
    

    text = re.sub(r'——\s?——', '', text) # replace --   -- to ''
    text = re.sub(r'\s+', ' ', text)    # strip all whitespaces
    
    text = re.sub(r'，\s?，', '，', text)  # merge commas separating only whitespace
    text = re.sub(r'，\s?。', '。', text) # , . -> ,
    #text = re.sub(r'？\s?。', '？', text)# ? . -> ?
    text = re.sub(r'\s+', ' ', text)    # strip all redundant whitespace that could have been caused by preprocessing
    
    #text = re.sub(r'\s+？', '？', text)
    text = re.sub(r'\s+，', '，', text)
    text = re.sub(r'。[\s+。]+', '。 ', text)
    text = re.sub(r'\s+。', '。 ', text)
    
    #text = re.sub(r'？\s+', '？', text)
    #text = re.sub(r'，\s+', '，', text)
    #text = re.sub(r'。\s+', '。 ', text)
    
    return text.strip().lower()

datasets = [[clean_text(text) for text in ds] for ds in datasets]

print([len([t for t in ds if len(t)>0]) for ds in datasets]) # remove all 0 word datasets

print([len(' '.join(ds).split(' ')) for ds in datasets]) # make them sentences separated by a space for tokenizing

tokenizer = AutoTokenizer.from_pretrained(model_type)


target_ids = tokenizer.encode("。，")[1:-1]
print(tokenizer.convert_ids_to_tokens(target_ids))


target_token2id = {t: tokenizer.encode(t)[-2] for t in "。，"}
print(target_token2id)



target_ids = list(target_token2id.values())
print(target_token2id.items())
#target_ids

import jieba
id2target = {
    0: 0,
    -1: -1,
}
for i, ti in enumerate(target_ids):
    id2target[ti] = i+1
target2id = {value: key for key, value in id2target.items()}
# print(id2target, target2id)

def create_target(text):
    encoded_words, targets = [], []
    
    words = list(jieba.cut(text,HMM=True)) ## ignore the first space
    words2 = []
    for i in range(len(words)):
        encoded_word = tokenizer.encode(words[i])
        #print(words[i],encoded_word)
        if (len(encoded_word[1:-1]) > 1 and encoded_word[1] != 6) or (len(encoded_word[1:-1]) > 2 and encoded_word[1] == 6):
            for word in encoded_word[1:-1]:
                if word != 6:
                    encoded_words.append(word)
                    targets.append(-1)
            targets = targets[:-1]   
        elif len(encoded_word[1:-1]) == 0:
            continue
        else:
            #print("Here! ",encoded_word)
            s = 2 if encoded_word[1] == 6 else 1
            encoded_words.append(encoded_word[s])
                
            
            
        if words[i] not in ["。","，"," ","▁"]:
            if i < len(words) -1 and words[i+1] in ["。","，"," ","▁"]:
                ##words2.append(words[i])
                targets.append(0)
                pass
            else:
                targets.append(0)
                encoded_words.append(6)
                targets.append(0)
        else:
            if words[i] in ["▁"," "]:
                if i > 0 and words[i-1] not in ["。","，"," ","▁"]:
                    #encoded_words.append(" ")
                    #targets.append(0)
                    pass
            else:
                #print("YES",words[i])
                targets.append(id2target[target_token2id[words[i]]])
                # words2.append(words[i])
    
    encoded_words = [tokenizer.cls_token_id or tokenizer.bos_token_id] +\
                    encoded_words +\
                    [tokenizer.sep_token_id or tokenizer.eos_token_id]
    
    targets = [-1]+ targets + [-1]    
    
    return encoded_words, targets


print(id2target)
s = "谁能猜一猜：你大脑里神经元的总长有多少？ ”西班牙厨师被控告……“ 非常坚硬的土地。西班牙厨师被控告"
#s = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
#s = "算所， 日本"
#print(s)
s = clean_text(s)
#print(s)
data, targets = create_target(s)
#print(data)
#print(targets)
print([(tokenizer._convert_id_to_token(d), ta) for d,ta in zip(data[2:-1], targets[2:-1])])


#no split
encoded_texts, targets = [], []

for ds in datasets:
    x = list(zip(*(create_target(ts) for ts in tqdm(ds))))
    encoded_texts.append(x[0])
    targets.append(x[1])

def return_counts(encoded_texts, targets):
    # encoded_words, targets
    comma_count = 0
    word_count = 0
    #q_count = 0
    p_count = 0
    space_count = 0
    for target in targets:
        for tar in target:
            for ta in tar:
                comma_count += 1 if (ta == 2) else 0
                #q_count += 1 if (ta == 2) else 0
                p_count += 1 if (ta == 1) else 0
    sc = 0
    mwc = 0
    for text,target in zip(encoded_texts, targets):
        for tex,tar  in zip(text,target):
            en = 0
            for t,ta in zip(tex,tar):
                if t not in [6,5,0,-1,1,2,4] and ta != -1:
                    word_count+=1
                    en+=1
                elif t == 6 and ta != -1: # space
                    space_count+=1
                elif t in [5]:
                    mwc*=sc
                    sc += 1
                    mwc += en
                    mwc /= sc
                    en = 0
    return space_count, p_count, comma_count

os.makedirs(os.path.join(path, 'sgh', model_type), exist_ok=True)
space_count, p_count, comma_count = return_counts(encoded_texts,targets)

for i, name in enumerate(('train', 'valid', 'test')):
    with open(os.path.join(path, 'sgh') +'/'+ f'{model_type}/{name}_data.pkl', 'wb') as f: #create 2 new folders, one dataset, one xlm-roberta-base
        pickle.dump((encoded_texts[i], targets[i], space_count, p_count, comma_count), f)





