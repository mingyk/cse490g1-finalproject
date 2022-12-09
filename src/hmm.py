from src.preproc import conll_seq_generator
from src.constants import START_TAG, END_TAG, OFFSET, UNK
from src import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


# Deliverable 4.2
def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    
    weights = defaultdict(float)
    
    all_tags = list(trans_counts.keys())+ [END_TAG]
    
#     raise NotImplementedError
    for tag_from in trans_counts:
        for tag_to in all_tags:
            if tag_to == START_TAG:
                weights[(tag_to,tag_from)] = -np.inf
            else:
                weights[(tag_to,tag_from)] = np.log(trans_counts[tag_from][tag_to] + smoothing) - np.log(sum(trans_counts[tag_from].values()) + (len(all_tags) * smoothing))
    
    
    return weights


# Deliverable 3.2
def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), 0.0)
    
#     raise NotImplementedError
    for tag in tag_to_ix:
        for tag2 in tag_to_ix:
            value = -np.inf
            if (tag,tag2) in hmm_trans_weights:
                value = hmm_trans_weights[(tag,tag2)]
            tag_transition_probs[tag_to_ix[tag],tag_to_ix[tag2]] = value
        for vocab in word_to_ix:
            value = 0
            if tag == START_TAG or tag == END_TAG:
                value = -np.inf
            if (tag,vocab) in nb_weights:
                value = nb_weights[(tag,vocab)]
#             print(tag,vocab,value)
            emission_probs[word_to_ix[vocab]][tag_to_ix[tag]] = value
    
    
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
