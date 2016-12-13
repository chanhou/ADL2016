
#python translate.py --data_dir ./data --data_dir_train ./data/train --data_dir_test ./data/valid --train_dir ./model --en_vocab_size=40000 --fr_vocab_size=40000 --final ./data/test.en --write ./model/output
#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=0B_9c-5D_LAa_OW1ZVmxqWHltaUU" -O ./model_translate/translate.ckpt-16200
#python gdown.py 0B_9c-5D_LAa_OW1ZVmxqWHltaUU ./model_translate/translate.ckpt-16200
wget -O ./model_translate/translate.ckpt-16200 https://mslab.csie.ntu.edu.tw/~chanhou/translate.ckpt-16200
python translate.py --data_dir ./data_translate --data_dir_train ./data_translate/train --data_dir_test ./data_translate/valid --train_dir ./model_translate --en_vocab_size=40000 --fr_vocab_size=40000 --final $1 --write $2 --decode=True
