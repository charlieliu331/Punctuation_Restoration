'''
1) removes the time and coversation name stamp.

2) punctuation symbols to punctuation.
    <c/> -> ,      <s/> -> .

'''

import argparse
import os
parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--data-path',type=str,default='',help='path to the directory that stores sgh_train,test and val')



def symbol_2_punc(sentence):
    cleaned = sentence.replace('<c/>','，')
    cleaned = cleaned.replace('<s/>', '。')
    cleaned = cleaned.replace(' ，','，')
    cleaned = cleaned.replace(' 。','。')

    return cleaned

args = parser.parse_args()
path = args.data_path


with open(os.path.join(args.data_path,"sgh_train.txt"), 'r') as f:
    train_text = f.readlines()

cleaned_text = [symbol_2_punc(' '.join(line.split(' ')[3:])) for line in train_text]
with open(os.path.join(args.data_path,"cleaned_sgh_train.txt"), 'w') as f:
    for l in cleaned_text:
        f.write(l)


with open(os.path.join(args.data_path,"sgh_test.txt"), 'r') as f:
    test_text = f.readlines()

cleaned_text = [symbol_2_punc(' '.join(line.split(' ')[3:])) for line in test_text]
with open(os.path.join(args.data_path,"cleaned_sgh_test.txt"), 'w') as f:
    for l in cleaned_text:
        f.write(l)


with open(os.path.join(args.data_path,"sgh_val.txt"), 'r') as f:
    val_text = f.readlines()

cleaned_text = [symbol_2_punc(' '.join(line.split(' ')[3:])) for line in val_text]
with open(os.path.join(args.data_path,"cleaned_sgh_val.txt"), 'w') as f:
    for l in cleaned_text:
        f.write(l)

