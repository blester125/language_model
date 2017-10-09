"""Create Zipf and heap graphs for talk."""

# [ Imports ]
# [ -Python ]
from collections import Counter
# [ -Third Party ]
import numpy as np
from nltk.corpus import brown, reuters, treebank, inaugural
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def count(corpus):
    return Counter([word.lower() for word in corpus.words()])

def zipf(i, corpus, ax, corpus_name):
    """Generate Zipf plots.

    Plots the frequency vs the frequency rank on a log log scale.
    """
    counter = count(corpus)

    counts = [token[1] for token in counter.most_common()]
    tokens = [token[0] for token in counter.most_common()]
    ranks = np.arange(1, len(counts)+1)

    log_counts = np.log10(counts)
    log_ranks = np.log10(ranks)

    clf = LinearRegression()
    clf.fit(log_ranks.reshape(-1, 1), log_counts.reshape(-1, 1))

    best_fit_line = clf.predict(log_ranks.reshape(-1, 1))

    ax.scatter(log_ranks, log_counts, c='k', s=.5)
    ax.plot(log_ranks, best_fit_line, c='r', linewidth=.5)

    ax.set_title("{} corpus".format(corpus_name))
    if i == 0:
        ax.set_ylabel("Absolute frequency of token")
        ax.set_xlabel("Frequency Rank of the token")

    return ax


def heaps(i, corpus, ax, corpus_name):
    """Generate the Heap plots.

    Plot the vocabulary size vs the number of documents.
    """
    vocab_size = []
    vocab = set()
    for fileid in corpus.fileids():
        for word in corpus.words(fileids=[fileid]):
            vocab.add(word)
        vocab_size.append(len(vocab))

    log_counts = np.log10(vocab_size)
    log_number_of_docs = np.log10(np.arange(1, len(corpus.fileids()) + 1))

    clf = LinearRegression()
    clf.fit(log_number_of_docs.reshape(-1, 1), log_counts.reshape(-1, 1))

    best_fit_line = clf.predict(log_number_of_docs.reshape(-1, 1))

    ax.scatter(log_number_of_docs, log_counts, c='k', s=.5)
    ax.plot(log_number_of_docs, best_fit_line, c='r', linewidth=.5)

    if i == 0:
        ax.set_ylabel("Vocabulary Size")
        ax.set_xlabel("Number of Docs")


fig, axes = plt.subplots(2, 4, figsize=(15, 10))
fig.suptitle("Zipf and Heaps Laws", fontsize=20)
corpui = [brown, reuters, treebank, inaugural]
names = ['Brown', 'Reuters', 'Penn Treebank', 'Inaugural']

for i, (c, ax, name) in enumerate(zip(corpui, axes[0], names)):
    zipf(i, c, ax, name)

for i, (c, ax, name) in enumerate(zip(corpui, axes[1], names)):
    heaps(i, c, ax, name)

plt.savefig("Zipf_and_Heap.png")
