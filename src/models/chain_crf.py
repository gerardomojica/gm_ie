#!/bin/bash
# Luis Mojica - June. 2015
# Information Extraction - Opinion Mining - Graphical Model
# The University of Texas at Dallas
# Linear Chain CRF with open gm

import sys
import getopt
# from ..template import featurize
# This works in this way because the file is called as a module!
from template.featurize import featurize_file
import numpy as np
import opengm
from scipy.sparse import lil_matrix, csr_matrix
from scipy import sparse

# Define the number of features:
n_features = 1000

# Process the features
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=n_features,input_type='string',non_negative=True)

# Need global variables:
# global w_vec
# global templated_data

# Define the python function for the unary factors:

def myFunc(labels):
    p=1.0
    s=0
    for l in labels:
        p*=l
        s+=l
    return p+s


# pf=opengm.PythonFunction(myFunc,[2,2])
# print pf

def unary_obs(label):
    # print len(templated_data)
    # lil_matrix
    # features = csr_matrix

    p=1.0
    s=0
    for l in label:
        p*=l
        s+=l
    return p+s

def function_creator(name, obs_feat_idx):

    def foo(label):

        # feature list:
        feat_list = []

        # Add the tag to every feature:
        for f_val in obs_feat_idx['F']:
            feat_list.append(label+'~'+f_val)

        fv = fh.transform([feat_list])
        # Compute the dot product with the weights vector:
        # print fv.T.dot(w_vec.tocsr())
        return fv.dot(w_vec)
        
        # exit()

    # Give a name to the function
    foo.__name__ = name
    return foo


# Create the unary functions:
def create_unaries(sentence_data, gm):
    numVar = len(sentence_data)

    # Go over the length of the sentence:
    for idx in xrange(numVar):
        # obs_feat_idx = sentence_data[idx]['F']
        obs_feat_idx = sentence_data[idx]
        y_tag = obs_feat_idx['y']
        
        # Create the name of the function
        f_name = 'f_'+str(idx)
        f_name = function_creator(f_name, obs_feat_idx)

        # Add the function as a factor to the pgm:
        py_func = opengm.PythonFunction(f_name,[2])
        gm_func = gm.addFunction(py_func)
        gm.addFactor(gm_func,idx)
        # gm.addFactor(opengm.PythonFunction(f_name,[2]),idx)
        # gm.addFactor(gm.addFunction(f_name),idx)
        # exit()
        # print f_name(y_tag)
        return 'hey','hou'

# Instantiate the current sentence's gm:
def instantiate_sentence(sentence_data):
    # Get the number of tokens:
    numVar = len(sentence_data)

    # Binary variables: [2]
    gm=opengm.gm([2*numVar,numVar])

    # Create the unary factors for the pgm
    gm, factor_list = create_unaries(sentence_data, gm)

    # # Use observations as unary factors:
    # pf=opengm.PythonFunction(unary_obs,[2,2])
    # print pf

# Train the model's parameters:
def train(templated_data):

    # Transform the weights vector:
    global w_vec
    w_vec = sparse.csr_matrix(w_vec).T

    # Token counter:
    token_counter = 0
    # Go over each of the sentences and build the modle:
    for sent_id, sentence in enumerate(templated_data):
        # Get the number of tokens:
        # num_tokens = len(templated_data[sent_id])
        # print num_tokens
        # Build the graphical model:
        gm = instantiate_sentence(templated_data[sent_id])
        # create_unaries(templated_data[sent_id], gm)
        exit()

        # Keep track of the token counter:
        # token_counter += num_tokens

def main(argv):
    options, remainder = getopt.getopt(argv, 'o:v', ['train_file=',])

    # Parse the arguments
    for opt, arg in options:
        if opt == '--train_file':
            train_file = arg
            # print train_file

    # Featurize the train_file:
    # templated_data, hashed_feature_matrix = featurize_file(train_file)
    global templated_data
    templated_data = featurize_file(train_file)

    # Initialize the sparse weights vector:
    global w_vec

    # w_vec = lil_matrix((1,n_features))
    # w_vec = np.ones((1,n_features))
    w_vec = np.full((1,n_features),0.1)
    # w_vec[0,100]=45
    # w_vec = sparse.csr_matrix(w_vec)
    # exit()

    # Debug:
    # Sentence 0, token 1's features
    # print templated_data[0][1]['F']
    # print templated_data[0][1]
    # Sentence 0, token 1's hashed features
    # print hashed_feature_matrix
    # exit()

    # Train:
    # train(templated_data, hashed_feature_matrix)
    train(templated_data)


if __name__ == '__main__':
    main(sys.argv[1:])
