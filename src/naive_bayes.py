from src.constants import OFFSET
from src import clf_base, evaluation, preproc

import numpy as np
from collections import defaultdict

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preproc.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
#     print(token_level_docs)
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights


# Can copy from A1
def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    counts = defaultdict(float)
    for i in range(len(y)):
        if y[i] == label:
            counter = x[i]
            for word in counter:
                if word not in counts:
                    counts[word] = 0
                counts[word] += counter[word]
    return counts
#     raise NotImplementedError



# Can copy from A1
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    pxy = defaultdict(float)
    filtered = get_corpus_counts(x,y,label)
#     print(filtered)
    for (word,_) in vocab:
        count = 0
        if word in filtered:
            count = filtered[word]
        pxy[word] = np.log(count + smoothing) - np.log(sum(filtered.values()) + (len(vocab) * smoothing))
    return pxy


# Can copy from A1
def estimate_nb(x,y,smoothing):
    """
    estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    weights = defaultdict(float)
    vocab = set([])
    for doc in x:
        vocab.update(set(doc.items()))
    label_counts = defaultdict(float)
    for label in y:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    for label in set(y):
        logprobs = estimate_pxy(x,y,label,smoothing,vocab)
        weights.update(clf_base.make_feature_vector(logprobs,label))
        weights.update({(label, OFFSET): np.log(label_counts[label] / len(y))})
    return weights
#     raise NotImplementedError

    

# Can copy from A1
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    best_value = 0
    scores = {}
    best_accuracy = 0
    for smoother in smoothers:
        y_hat = clf_base.predict_all(x_dv,estimate_nb(x_tr,y_tr,smoother),set(y_tr))
        acc = evaluation.acc(y_hat, y_dv)
        if acc > best_accuracy:
            best_accuracy = acc
            best_value = smoother
        scores[smoother] = acc
    return best_value, scores
#     raise NotImplementedError

