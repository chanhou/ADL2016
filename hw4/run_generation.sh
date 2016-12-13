
#python generate.py --data_dir ./data --data_dir_train ./data/train --data_dir_test ./data/valid --train_dir ./model --en_vocab_size=40000 --fr_vocab_size=40000 --final ./data/test.txt --write ./model/output --size=300
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
#python gdown.py 0B_9c-5D_LAa_Y1Jic05ISFVTOTA ./model_generate/translate.ckpt-57200
wget -O ./model_generate/translate.ckpt-57200 https://mslab.csie.ntu.edu.tw/~chanhou/translate.ckpt-57200
python generate.py --data_dir ./data_generate --data_dir_train ./data_generate/train --data_dir_test ./data_generate/valid --train_dir ./model_generate --en_vocab_size=40000 --fr_vocab_size=40000 --final $1 --write $2 --size=300 --decode=True
