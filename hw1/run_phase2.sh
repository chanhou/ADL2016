!/bin/bash

mkdir $2
mkdir tmp

python src/word2vec_optimized.py --train_data=$1  --eval_data=word2vec/questions-words-phase2-dev.txt  --save_path=tmp --epochs_to_train=10 --embedding_size=250 --concurrent_steps=8

python filterVocab/filterVocab.py filterVocab/fullVocab_phase2.txt < tmp/filter_word2vec.txt > $2/filter_vec.txt

rm -r tmp
