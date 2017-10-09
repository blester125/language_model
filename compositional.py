"""Language Model with Character Composition for embedding."""

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

num_chars = len(char_vocab)
num_words = len(word_vocab)

# Dynet Model
model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)
trainer.learning_rate = learning_rate

CHAR_EMBEDDING_MATRIX = model.add_lookup_parameters((num_chars, char_embedding_size))

word_lstm = dy.LSTMBuilder(num_layers, embedding_size, lstm_out, model)
char_fw_lstm = dy.LSTMBuilder(1, char_embedding_size, char_lstm_out, model)
char_bw_lstm = dy.LSTMBuilder(1, char_embedding_size, char_lstm_out, model)

softmax_w = model.add_parameters((num_words, lstm_out))
softmax_b = model.add_parameters((num_words))

params = [CHAR_EMBEDDING_MATRIX, softmax_w, softmax_b]
params.extend(*word_lstm.get_parameters())
params.extend(*char_fw_lstm.get_parameters())
params.extend(*char_bw_lstm.get_parameters())
print("Number of Params: {}".format(sum(np.prod(p.shape()) for p in params)))

def word_rep(word, fw_init, bw_init):
    pad = char_vocab.get('<*>')
    indices = [pad] + [char_vocab.get(c) for c in word] + [pad]
    embedded = [CHAR_EMBEDDING_MATRIX[i] for i in indices]
    forward = fw_init.transduce(embedded)
    backward = bw_init.transduce(embedded)
    return dy.concatenate([forward[-1], backward[-1]])

def calc_lm_loss(words):
    dy.renew_cg()

    sm_w = dy.parameter(softmax_w)
    sm_b = dy.parameter(softmax_b)

    w_init = word_lstm.initial_state()
    fw_init = char_fw_lstm.initial_state()
    bw_init = char_bw_lstm.initial_state()

    word_embeddings = [word_rep(w, fw_init, bw_init) for w in words]
    labels = [word_vocab.get(w) for w in words]

    s = w_init.add_input(word_rep(words[-1], fw_init, bw_init))

    losses = []
    for word_emb, label in zip(word_embeddings, labels):
        logits = sm_w * s.output() + sm_b
        loss = dy.pickneglogsoftmax(logits, label)
        losses.append(loss)
        s = s.add_input(word_emb)
    return dy.esum(losses)

train(trainer, calc_lm_loss, num_epochs)
