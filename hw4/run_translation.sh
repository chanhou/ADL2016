
#python translate.py --data_dir ./data --data_dir_train ./data/train --data_dir_test ./data/valid --train_dir ./model --en_vocab_size=40000 --fr_vocab_size=40000 --final ./data/test.en --write ./model/output
python translate.py --data_dir ./data_translate --data_dir_train ./data_translate/train --data_dir_test ./data_translate/valid --train_dir ./model_translate --en_vocab_size=40000 --fr_vocab_size=40000 --final $1 --write $2 --decode=True
