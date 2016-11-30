import sys
import os

#os.system('cat atis.train.w-intent.iob | shuf > atis.train')
'''
line_count = 0
with open('atis.train','r')as f:
    for line in f:
        line_count += 1

training = int(line_count*0.8)

with open('atis.train','r')as f, open('rnn-nlu/data/atis/train/train.label','w') as trainL, open('rnn-nlu/data/atis/train/train.seq.in','w') as trainI, open('rnn-nlu/data/atis/train/train.seq.out','w') as trainO, open('rnn-nlu/data/atis/valid/valid.label','w') as validL, open('rnn-nlu/data/atis/valid/valid.seq.in','w') as validI, open('rnn-nlu/data/atis/valid/valid.seq.out','w') as validO:
    for ind,line in enumerate(f):
        line = line.split()
        for index,word in enumerate(line):
            if word=='EOS':
                if len(line[-1].split('#'))>1: line[-1] = line[-1].split('#')[0]
                if line[-1]=='atis_cheapest': line[-1] = 'atis_airfare'
                if ind< training:
                    trainI.write(' '.join(line[1:index])+'\n')
                    trainO.write(' '.join(line[index+2:len(line)-1])+'\n')
                    trainL.write(line[-1]+'\n')
                else:
                    validI.write(' '.join(line[1:index])+'\n')
                    validO.write(' '.join(line[index+2:len(line)-1])+'\n')
                    validL.write(line[-1]+'\n')
                break
'''

with open(sys.argv[1],'r')as f, open('rnn-nlu/data/atis/test/test.seq.in','w') as test:
    for line in f:
        line = line.split()
        test.write(' '.join(line[1:-1])+'\n')
