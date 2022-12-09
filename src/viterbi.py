import operator
from collections import defaultdict, Counter
from src.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Deliverable 3.3
def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    bptrs = []
    viterbivars=[]

    for cur_tag in list(all_tags):
        
#         raise NotImplementedError
        # v_t(j) = max(v_t-1(i)*a_ij*b_j(o_t)) --> add for log
        index = tag_to_ix[cur_tag]
        best_bp_index = argmax(prev_scores + transition_scores[index, :] + cur_tag_scores[index])
        bptrs.append(best_bp_index)
        viterbivars.append(prev_scores[0, best_bp_index] + transition_scores[index, best_bp_index] + cur_tag_scores[index])

    
    return viterbivars, bptrs


# Deliverable 3.4
def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    
    Hint: Pay attention to the dimension of cur_tag_scores. It's slightly different from the one in viterbi_step().
    """
    
    ix_to_tag = { v:k for k,v in tag_to_ix.items() }
    
    # setting all the initial score to START_TAG
    # remember that END_TAG is in all_tags
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1,-1)
    whole_bptrs = []
    for m in range(len(cur_tag_scores)):
        
#         raise NotImplementedError
        output = viterbi_step(all_tags, tag_to_ix, cur_tag_scores[m], transition_scores, prev_scores)
        whole_bptrs.append(output[1])
        
#         ls = np.array(output[0]).astype(np.float32)
        tc_ls = torch.tensor(output[0])#torch.from_numpy(ls)
        prev_scores = torch.autograd.Variable(tc_ls).view(1,-1)
        
#     prev_scores += 
    # after you finish calculating the tags for all the words: don't forget to calculate the scores for the END_TAG
    # bestpathpointer = argmax(viterbi[s,t])
    # bestpathprob = max(viterbi[s,t])
#     best_bp_index = argmax(prev_scores)
#     print(best_bp_index)
#     print(argmax(prev_scores + transition_scores))
#     print(prev_scores)
#     print(transition_scores)
#     print(argmax(transition_scores[tag_to_ix[END_TAG]].view(1,-1)))
#     print(prev_scores + transition_scores[tag_to_ix[END_TAG], :])
    best_bp_index = argmax(prev_scores + transition_scores[tag_to_ix[END_TAG], :])
#     print(best_bp_index)
#     best_bp_index = argmax(prev_scores + transition_scores[index, :] + cur_tag_scores[index])
#     print(end_tag)
    best_score = prev_scores[0, best_bp_index] + transition_scores[tag_to_ix[END_TAG], best_bp_index]
#     print(best_score)
    
    # Calculate the best_score and also the best_path using backpointers and don't forget to reverse the path
    # bestpath = start with bestpathpointer, follows backpointer[] back in time
    path_score = best_score
    best_path = []
    whole_bptrs = whole_bptrs[::-1]
    for bp in whole_bptrs:
        best_path.append(ix_to_tag[best_bp_index])
        best_bp_index = bp[best_bp_index]
    best_path = best_path[::-1]
    
    
    return path_score, best_path
