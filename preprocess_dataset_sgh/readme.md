```
$ pip install -r ./preprocess_dataset_sgh/requirements.txt
$ python3 path/to/TextGrids_to_txts.py --data-path path/to/textgrid/directory
$ . path/to/combine_txts.sh --path path/to/sgh_txt
$ python3 path/to/txtremoveEMPTY.py --data-path path/to/directory/with/all.txt
$ python3 path/to/train_test_val_split.py --data-path path/to/directory/with/cleaned_all.txt
$ python3 path/to/punc_sym_to_punc.py --data-path path/to/directory/with/sgh_train/test/val.txt
$ python3 path/to/create_pkl_dataset_new.py --data-path path/to/directory/with/cleaned_sgh_train/test/val.txt


$ cd /new_Multilingual-Sentence-Boundary-detection/
#training
$ python3 ./src/main.py --num-epochs 10 --train-data-path /sgh/xlm-roberta-base/ --val-data-path sgh/xlm-roberta-base/ --model-path /punctuator-model/sgh_10epoch- --eval-type valid/tst
#testing
$ python3 ./src/main.py --train-data-path /sgh/xlm-roberta-base/ --val-data-path /sgh/xlm-roberta-base/ --action val --model-path /path/to/model/dir/ --stage sgh_10epoch-xlm-roberta-base-epoch-1.pth --eval-type valid/tst
```

sample code
```
$ pip install -r ./preprocess_dataset_sgh/requirements.txt
$ cd sgh_dataset/
$ python3 ../preprocess_dataset_sgh/TextGrids_to_txts.py --data-path ./sgh_TextGrid
$ . ../preprocess_dataset_sgh/combine_txts.sh --path ./sgh_txt
$ python3 ../../preprocess_dataset_sgh/txtremoveEMPTY.py --data-path ../
$ python3 ../../preprocess_dataset_sgh/train_test_val_split.py --data-path ../
$ python3 ../../preprocess_dataset_sgh/punc_sym_to_punc.py --data-path ../
$ python3 ../../preprocess_dataset_sgh/create_pkl_dataset_new.py --data-path ../

#use GPU
$ cd /new_Multilingual-Sentence-Boundary-detection/
#training
$ python3 ./src/main.py --num-epochs 10 --train-data-path /sgh/xlm-roberta-base/ --val-data-path sgh/xlm-roberta-base/ --model-path /punctuator-model/sgh_10epoch- --eval-type valid/tst
#testing
$ python3 ./src/main.py --train-data-path /sgh/xlm-roberta-base/ --val-data-path /sgh/xlm-roberta-base/ --action val --model-path /path/to/model/dir/ --stage sgh_10epoch-xlm-roberta-base-epoch-1.pth --eval-type valid/tst
```

The names of python scripts are unique, can change the relative path to files accordingly.
TextGrids_to_txts.py has been updated in the latest version, so please use the latest scripts.
