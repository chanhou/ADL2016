import sys
import tf_glove


corpus = []
with open(sys.argv[1], 'rb') as f:
    for line in f:
        for q in line.split():
            corpus.append(q)
print 'Done loading...'
print len(corpus)
print corpus[:10]

model = tf_glove.GloVeModel(embedding_size=128, context_size=10, min_occurrences=2, max_vocab_size=70000, learning_rate=0.7)
model.fit_to_corpus([corpus])

print 'Start to train...'
model.train(num_epochs=10)

#model.embeddings
#model.id_for_word
