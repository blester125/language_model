"""Language Model at the Word level."""

# [ Imports ]
# [ -Third Party ]
import dynet as dy
import numpy as np
# [ -Project ]
from utils import word_vocab
from utils import train
from utils import embedding_size
from utils import lstm_out
from utils import num_layers
from utils import num_epochs
from utils import learning_rate
from utils import count_params

num_words = len(word_vocab)

# Dynet Model
model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)
trainer.learning_rate = learning_rate

EMBEDDING_MATRIX = model.add_lookup_parameters((num_words, embedding_size))

lstm = dy.LSTMBuilder(num_layers, embedding_size, lstm_out, model)

softmax_w = model.add_parameters((num_words, lstm_out))
softmax_b = model.add_parameters((num_words))

params = [EMBEDDING_MATRIX, softmax_w, softmax_b]
params.extend(*lstm.get_parameters())
print("Number of Params: {}".format(count_params(params)))

def calc_lm_loss(words):
    dy.renew_cg()

    sm_w = dy.parameter(softmax_w)
    sm_b = dy.parameter(softmax_b)

    w_init = lstm.initial_state()

    word_ids = [word_vocab.get(w) for w in words]

    s = w_init.add_input(EMBEDDING_MATRIX[word_ids[-1]])

    losses = []
    for word_id in word_ids:
        logits = sm_w * s.output() + sm_b
        loss = dy.pickneglogsoftmax(logits, word_id)
        losses.append(loss)
        s = s.add_input(EMBEDDING_MATRIX[word_id])
    return dy.esum(losses)

train(model, trainer, calc_lm_loss, num_epochs, "models/word")
