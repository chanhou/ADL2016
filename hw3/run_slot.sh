!/bin/bash

python preprocess.py $1
python rnn-nlu/run_multi-task_rnn_slot.py --data_dir rnn-nlu/data/atis --train_dir rnn-nlu/why --max_sequence_length 50 --task joint --bidirectional_rnn True --output $2


