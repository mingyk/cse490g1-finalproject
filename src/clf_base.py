from src.constants import OFFSET
import numpy as np

import operator

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]


# Deliverable 2.1 - can copy from A1
def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    output = {}
    for key in base_features.keys():
        output[(label, key)] = base_features.get(key)
    output[(label, OFFSET)] = 1
    return output
#     raise NotImplementedError
    

# Deliverable 2.1 - can copy from A1
def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = {}
    for label in labels:
        scores[label] = 0
        for feat in base_features:                
            scores[label] += weights[(label, feat)] * base_features[feat]
        scores[label] += weights[(label, OFFSET)] * 1
    return argmax(scores), scores
#     raise NotImplementedError
