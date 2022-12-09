# Final Project

Name: Min Kim
UWNetID: mingyk
Date: December 12, 2022

### Abstract
This project explores methods on sequence labeling to implement a part-of-speech tagger on multiple languages including English, Korean, Greek, and Swedish Sign Language. The model will be built based on English data, but to evaluate the validity of it, I am training with multiple languages that are from the same linguistic family to a totally different family.

### Problem Space
Part of speech tagging, also known as POS, is "the process of marking up a word in a text as corresponding to a particular part of speech, based on both its definition and its context" in corpus linguistics. The recent development and spread of AI and ML usages have incorporated this idea of POS to dynamically improve the AI behavior of Natural Language Processing (NLP) in problem spaces such as Siri or Alexa. In this project, I am going to utilize multiple techniques to create the best model with the highest result in accuracy that could handle not only English, but also languages around the world.

### Related Work
Language corpus data are all gathered from Universal Dependencies. (Link: https://universaldependencies.org)

### Methodology
I utilized sequence labeling methods including Hidden Markov Models, GRU, BiGRU, BiLSTM, and BiLSTM-CRF to evaluate the effiency of each model on POS with multiple languages (English, Korean, Greek, and Swedish Sign Language).

### Experiments and Evaluation
Refer to main.ipynb under **Model Training** sections to see each dev, train, and test accuracies with plots of losses & accuracies.

### Results
#### Example Classifiers (All-Noun, Majority-Rule, Naive Bayes)
*All tested on English corpus only*
||All-Noun|Majority-Rule|Naive Bayes|
|---|---|---|---|
|Dev|0.1528|0.8111|0.8105|
|Train|0.1654|0.9284|0.9284|
|Test|0.1744|**0.7340**|0.7316|

#### Hidden Markov Model
||English|Greek|Korean|Swedish SL|
|---|---|---|---|---|
|Dev|0.8409|0.9022|0.8146|0.6418|
|Train|0.9622|0.9654|0.9699|0.9891|
|Test|0.7677|**0.9069**|0.8107|0.7518|

#### GRU
||English|Greek|Korean|Swedish SL|
|---|---|---|---|---|
|Dev|0.8574|0.8126|0.7512|0.5746|
|Train|0.9862|0.8456|0.7762|0.9923|
|Test|0.7436|**0.8103**|0.7411|0.5638|

#### BiGRU
||English|Greek|Korean|Swedish SL|
|---|---|---|---|---|
|Dev|0.8753|0.7875|0.7533|0.5775|
|Train|0.9887|0.8255|0.7795|0.9969|
|Test|0.7840|**0.7872**|0.7439|0.6135|

#### BiLSTM
||English|Greek|Korean|Swedish SL|
|---|---|---|---|---|
|Dev|0.8656|0.8829|0.7736|0.5716|
|Train|0.9792|0.9021|0.8032|0.9301|
|Test|0.7696|**0.8850**|0.7651|0.6099|

#### BiLSTM-CRF
||English|Greek|Korean|Swedish SL|
|---|---|---|---|---|
|Dev|0.8188|0.8944|0.7719|0.9953|
|Train|0.9721|0.8944|0.7718|0.9922|
|Test|0.7298|**0.8767**|0.7386|0.7340|

### Discussion and Example
Based on results, it seems like HMM Model had the highest test accuracy for all languages. However, based on their training data, it is likely that these models could've been overfitted. The model with the lowest training accuracy was GRU and BiGRU. Of all languages, Greek tend to have the highest accuracy results, which is interesting since English corpus was the biggest, then Greek, Korean, followed by Swedish SL. Also it is expected Swedish SL to have low accuracies because it has a significantly lower amount of data. One interesting point I figured was that results are unaffected by the linguistic family of each language.