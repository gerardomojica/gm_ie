#!/bin/bash
# Luis Mojica - June. 2015
# Information Extraction - Opinion Mining - Graphical Model
# The University of Texas at Dallas
# Contains the script calls

# ############################# 
# ############################# 
# ############################# Tests with the template features using the CRFsuite:

# Use the original tools from crf_suite on the conll task
cat ../corpus/train.txt | template/chunking.py > ../corpus/train.crfsuite.txt
# debug, dont use:
cd delete
cat ../../corpus/train.txt | chunking.py

cat ../corpus/crf_suite_conll.txt | template/chunking.py > ../corpus/train.crfsuite.txt

# Using my own template:
cat ../corpus/crf_suite_conll.txt | template/featurize.py

# Testing the sklearn feature handler:
python template/hashing_vs_dict_vectorizer.py

# ############################# 
# ############################# 
# ############################# Integrate the hashing trick and build a linear chain crf:

# Note that I'm calling the file as module
python -m models.chain_crf --train_file ../corpus/train.txt
# On the sample file:
python -m models.chain_crf --train_file ../corpus/crf_suite_conll.txt
# On a super small traning file, 2 sentences:
python -m models.chain_crf --train_file ../corpus/dummy_train.txt

# Stochastic Gradient Descent: /Users/luis/Dropbox/Luisn/PhD/Lab/Projects/gm_ie/src
python -m models.chain_crf2 --train_file ../corpus/dummy_train.txt


# ############################# 
# ############################# 
# ############################# Testing the creation of dynamic functions:

python -m delete.dynamic_functions


# ############################# 
# ############################# 
# ############################# Comming back on implementing the crf:

python -m models.multi_softmax

# ############################# 
# ############################# 
# ############################# Reemplementing featurization, softmax with graphical models:
cd /Users/luis/Dropbox/Luisn/PhD/Lab/Projects/gm_ie/src
python -m models.logistic_regression_gm
