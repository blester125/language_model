"""Language Model for back off to LSTM."""

# [ Imports ]
# [ -Third Party ]
import dynet as dy
import numpy as np
# [ -Project ]
from utils import char_vocab
from utils import word_vocab
from utils import train
from utils import char_embedding_size
from utils import embedding_size
from utils import char_lstm_out
from utils import lstm_out
from utils import num_layers
from utils import num_epochs
from utils import learning_rate
from utils import count_params

num_chars = len(char_vocab)
num_words = len(word_vocab)

# Dynet Model
model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)
trainer.learning_rate = learning_rate

CHAR_EMBEDDING_MATRIX = model.add_lookup_parameters((num_chars, char_embedding_size))
WORD_EMBEDDING_MATRIX = model.add_lookup_parameters((num_words, embedding_size))

word_lstm = dy.LSTMBuilder(num_layers, embedding_size, lstm_out, model)

char_fw_lstm = dy.LSTMBuilder(num_layers, char_embedding_size, char_lstm_out, model)
char_bw_lstm = dy.LSTMBuilder(num_layers, char_embedding_size, char_lstm_out, model)

softmax_w = model.add_parameters((num_words, lstm_out))
softmax_b = model.add_parameters(num_words)

params = [CHAR_EMBEDDING_MATRIX, WORD_EMBEDDING_MATRIX, softmax_w, softmax_b]
params.extend(*word_lstm.get_parameters())
params.extend(*char_fw_lstm.get_parameters())
params.extend(*char_bw_lstm.get_parameters())
print("Number of Params: {}".format(count_params(params)))

def word_vector(w, fw_init, bw_init):
    if word_vocab.counts[w] > 5:
        index = word_vocab.get(w)
        return WORD_EMBEDDING_MATRIX[index]
    else:
        pad = char_vocab.get('<*>')
        indices = [pad] + [char_vocab.get(c) for c in w] + [pad]
        embedded = [CHAR_EMBEDDING_MATRIX[i] for i in indices]
        forward = fw_init.transduce(embedded)
        backward = bw_init.transduce(reversed(embedded))
        return dy.concatenate([forward[-1], backward[-1]])

def calc_lm_loss(words):
    dy.renew_cg()

    sm_w = dy.parameter(softmax_w)
    sm_b = dy.parameter(softmax_b)

    w_init = word_lstm.initial_state()
    char_fw_init = char_fw_lstm.initial_state()
    char_bw_init = char_bw_lstm.initial_state()

    word_embeddings = [word_vector(w, char_fw_init, char_bw_init) for w in words]
    labels = [word_vocab.get(w) for w in words]

    s = w_init.add_input(WORD_EMBEDDING_MATRIX[word_vocab.get('<s>')])

    losses = []
    for word_emb, label in zip(word_embeddings, labels):
        logits = sm_w * s.output() + sm_b
        loss = dy.pickneglogsoftmax(logits, label)
        losses.append(loss)
        s = s.add_input(word_emb)
    return dy.esum(losses)

train(model, trainer, calc_lm_loss, num_epochs, "models/backoff")
