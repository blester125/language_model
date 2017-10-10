"""Utils for the Language Model Demo."""

# [ Imports ]
# [ -Python ]
import random
from itertools import chain
from collections import Counter
# [ -Third Party ]
import numpy as np

# Hyperparameters
num_layers = 1
char_embedding_size = 20
embedding_size = 256
lstm_out = 256
char_lstm_out = embedding_size // 2

learning_rate = 0.001
num_epochs = 5

class Vocab:
    def __init__(self, corpus):
        self.counts = Counter()
        for word in corpus:
            self.counts[word] += 1
        self.word_to_idx = {w: i for i, (w, _) in enumerate(self.counts.most_common())}
    def __len__(self):
        return len(self.word_to_idx)
    def get(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        else:
            return self.word_to_idx['<unk>']

def read(filename):
    sentences = []
    chars = set()
    with open(filename) as f:
        for line in f:
            sentence = line.strip().split()
            for word in sentence:
                chars.update(word)
            sentence.append("<s>")
            sentences.append(sentence)
    return sentences, chars


train_data, chars = read("data/penn/train.txt")
dev_data, _ = read("data/penn/valid.txt")
test_data, _ = read("data/penn/test.txt")

chars = list(chars)
chars.append("<*>")
chars.append("<s>")

word_vocab = Vocab(chain(*train_data))
char_vocab = Vocab(chars)


def train(model, trainer, forward, num_epochs, save_name):
    train_words = 0
    train_loss = 0
    min_dev_loss = 1000
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        for i, sentence in enumerate(train_data, 1):
            if i % 500 == 0:
                trainer.status()
                print(np.exp(train_loss / train_words))
                train_loss = 0
                train_words = 0
            if i % 10000 == 0 or i == len(train_data) - 1:
                dev_loss = 0
                dev_words = 0
                for dev_sentence in dev_data:
                    loss_exp = forward(dev_sentence)
                    dev_loss += loss_exp.scalar_value()
                    dev_words += len(dev_sentence)
                print("Dev Perplexity: {:.4f}".format(np.exp(dev_loss / dev_words)))
                if min_dev_loss > (dev_loss / dev_words):
                    min_dev_loss = (dev_loss / dev_words)
                    model.save(save_name + str(epoch) + "_" + str(i))

            loss_exp = forward(sentence)
            train_loss += loss_exp.scalar_value()
            train_words += len(sentence)
            loss_exp.backward()
            trainer.update()
        print("Epoch: {} finished.".format(epoch + 1))
    test_loss = 0
    test_words = 0
    for test_sentence in test_data:
        loss_exp = forward(test_sentence)
        test_loss += loss_exp.scalar_value()
        test_words += len(test_sentence)
    print("Test Perplexity: {:.4f}".format(np.exp(test_loss / test_words)))

def count_params(param_list):
    return sum(np.prod(p.shape()) for p in param_list)
