#!/bin/bash
# Luis Mojica - June. 2015
# Information Extraction - Opinion Mining - Graphical Model
# The University of Texas at Dallas
# Linear Chain CRF with open gm.
# Implement SGD Oct. 2015

import sys
import getopt
# from ..template import featurize
# This works in this way because the file is called as a module!
from template.featurize import featurize_file
import numpy as np
import opengm
from scipy.sparse import lil_matrix, csr_matrix
from scipy import sparse
import timeit
import math
import matplotlib.pyplot as plt

# Define the number of features:
n_features = 20

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
# print pf.shape
# print pf

# for c in opengm.shapeWalker(pf.shape):
#     print c," ",pf[c]

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

    def foo(label_idx):

        # Get the label string from the input name vector idx
        label = y_list[label_idx[0]]
        
        # feature list:
        feat_list = []

        # Add the tag to every feature:
        for f_val in obs_feat_idx['F']:
            feat_list.append(label+'~'+f_val)

        fv = fh.transform([feat_list])
        
        # Compute the dot product with the weights vector:
        # Log-linear = exp(∑fv)
        return math.exp(fv.dot(w_vec).toarray().item(0))

    # Give a name to the function
    foo.__name__ = name
    return foo

def second_order_function(name, obs_feat_idx=None):

    def foo(labels):

        # Unpack
        label_0_idx, label_1_idx = labels
        
        # Get the label string from the input name vector idx
        label_0 = y_list[label_0_idx]
        label_1 = y_list[label_1_idx]

        # The list of labels:
        # print y_list
        
        # feature list:
        feat_list = []

        # Add the tag to every feature:
        feat_list.append(label_0+'~'+label_1)

        # Get the feature number:
        fv = fh.transform([feat_list])
        
        # Compute the dot product with the weights vector:
        # Log-linear = exp(∑fv)
        return math.exp(fv.dot(w_vec).toarray().item(0))

    # Give a name to the function
    foo.__name__ = name
    return foo

# Get the list of features associated with a sample:
def sample_features(name, obs_feat_idx):
    
    def foo(label_idx):
        
        # Get the label string from the input name vector idx
        label = y_list[label_idx[0]]
        
        # feature list:
        feat_list = []

        # Add the tag to every feature:
        for f_val in obs_feat_idx['F']:
            feat_list.append(label+'~'+f_val)

        fv = fh.transform([feat_list])
        
        # Compute the dot product with the weights vector:
        # print fv, [feat_list]
        # print
        # print fv.dot(w_vec).toarray().item(0)
        # Non-log-linear
        # return fv.dot(w_vec).toarray().item(0)
        # Log-linear = exp(∑fv)
        return math.exp(fv.dot(w_vec).toarray().item(0))
        
        # exit()

    # Give a name to the function
    foo.__name__ = name
    print name(0)
    exit()
    return foo


# Create the unary functions:
def create_first_order(gm, sentence_data):
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
        # print py_func.shape
        # exit()
        gm_func = gm.addFunction(py_func)
        gm.addFactor(gm_func,idx)
        # print py_func[[0]]

    return gm


def create_second_order(gm, sentence_data, shared=True):
    numVar = len(sentence_data)

    # For shared variables:
    if shared:
        f_name = 'f_2nd_shared'
        f_name = second_order_function(f_name, None)
        py_func = opengm.PythonFunction(f_name, [2,2])
        gm_func = gm.addFunction(py_func)

    # Go over the length of the sentence:
    for idx in xrange(numVar-1):

        if not shared:
        
            # Create the name of the function (not shared!)
            f_name = 'f_'+str(idx)
            f_name = second_order_function(f_name, None)
            # print f_name
            # exit()

            # Add the function as a factor to the pgm:
            py_func = opengm.PythonFunction(f_name, [2,2])
            # print py_func.shape
            # print py_func[0,0]
            # exit()
            gm_func = gm.addFunction(py_func)
            gm.addFactor(gm_func,[idx,idx+1])

        else:
            gm.addFactor(gm_func,[idx,idx+1])


    return gm

# Instantiate the current sentence's gm:
def instantiate_sentence(sentence_data):
    # Get the number of tokens:
    numVar = len(sentence_data)
    # print 'numVar', numVar

    # The domain of the variables is the len of th y_list
    # Use the multiplier, the adder doesn't seem to work
    gm=opengm.gm([len(y_list)]*numVar,'multiplier')
    # gm=opengm.gm([len(y_list)]*numVar,'adder')
    # print gm.numberOfVariables
    # exit()

    # Create the 1st factors
    gm = create_first_order(gm, sentence_data)
    # print gm.factors()
    # for fac in gm.factors():
    #     print fac   

    # Create the 2nd order factors (True for shared factors)
    gm = create_second_order(gm, sentence_data, True)
    # print gm.factors()
    # for fac in gm.factors():
    #     print fac   

    # exit()



    return gm 
    
# # Compute the gradient for the current training sentence
# def gradient(gm, sentence_data):
#     numVar = len(sentence_data)

#     # Transform to lil:
#     global w_vec
#     w_vec = w_vec.tolil()
    
#     # Go over the length of the sentence:
#     for idx in xrange(numVar):

#         # obs_feat_idx = sentence_data[idx]['F']
#         obs_feat_idx = sentence_data[idx]
#         y_tag = obs_feat_idx['y']

#         # Get the label string from the input name vector idx
#         label = y_tag
        
#         # feature list:
#         feat_list = []

#         # Add the tag to every feature:
#         for f_val in obs_feat_idx['F']:
#             feat_list.append(label+'~'+f_val)

#         # Get the features idxs corresponding to the observations and y, f(x,y)
#         fv = fh.transform([feat_list])
#         rows,cols = fv.nonzero()
#         print cols

#     #     # Go over each of these indices in the weights vector:
#     #     for w_idx in cols:
#     #         w_vec[w_idx] = 0.1

#     # print w_vec

# Train the model's parameters:
def train(max_iter, templated_data):

    # Transform the weights vector:
    global w_vec
    w_vec = sparse.csr_matrix(w_vec).T

    # Token counter:
    token_counter = 0

    # Iterate up to max_iter:
    for x in xrange(max_iter):

        # Define learning rate based on the current iteration:
        # Not for now:
        eta = 0.5

        # Go over each of the sentences and build the modle:
        for sent_id, sentence in enumerate(templated_data):
            # Get the number of tokens:
            # num_tokens = len(templated_data[sent_id])

            # Build the graphical model:
            # gm = instantiate_sentence(templated_data[sent_id])
            gm = instantiate_sentence(sentence)
            # print gm
            # exit()

            # # visualize gm  
            # opengm.visualizeGm( gm,show=False,layout='spring',plotFunctions=True,
            #         plotNonShared=True,relNodeSize=0.4)
            # plt.savefig("chain_shared.png",bbox_inches='tight',dpi=300)  
            # plt.close()

            # # Run inference:
            # # The MAP
            bp = opengm.inference.BeliefPropagation(gm, 'integrator')
            bp.infer()
            # Get a list of all variables in the sentence:
            var_idx = [x for x in xrange(gm.numberOfVariables)]
            marg = bp.marginals(var_idx)
            print 'marg:\n',marg
            # arg = bp.arg()
            # print arg
            # print gradient(gm, sentence)
            # exit()


            # Keep track of the token counter:
            # token_counter += num_tokens
            # print 'aqui'

# # Implement stochatic gradient descent for general conditional models:
# def SGD(max_iter, templated_data):
    
#     # Initialize the sparse weights vector:
#     global w_vec
#     w_vec = np.zeros((1,n_features))

#     # Iterate up to max_iter:
#     for x in xrange(max_iter):

#         # Define learning rate based on the current iteration:
#         # Not for now:
#         eta = 0.5

#         # Go over each of the training samples:
#         for m in templated_data:

#             # Instantialte the graphical model:
#             m_sample = instantiate_sentence(m)
#             # print m_sample
#             bp = opengm.inference.BeliefPropagation(m_sample, 'integrator')
#             # bp.infer()
#             # marg = bp.marginals([0,1,2,3])
#             # print 'marg',marg
#             exit()

#         # exit()

#         #     # Get features used in m:
#         #     m_features = []

#         #     # Go over each of the features used in m:
#         #     for i in m_features:

#         #         # Get the current weight_i:
#         #         w_i = w_vec[i]

#         #         # # Compute the gradient from this sample m:
#         #         # g = 

#         #         # Update weight:
#         #         # w_i += eta*(g)

#         #         # Store the new w_i:
#         #         # w_vec[i] = w_i


def main(argv):
    options, remainder = getopt.getopt(argv, 'o:v', ['train_file=',])

    # Parse the arguments
    for opt, arg in options:
        if opt == '--train_file':
            train_file = arg

    # Featurize the train_file:
    global templated_data
    global y_list
    templated_data, y_list = featurize_file(train_file)

    # Initialize the sparse weights vector:
    global w_vec
    w_vec = np.zeros((1,n_features))
    # w_vec[0,13] = 1
    # w_vec[0,18] = 1
    # w_vec[0,7] = 1
    # w_vec[0,11] = 1
    # w_vec[0,100]=45

    # Train:
    # train(templated_data, hashed_feature_matrix)
    train(1, templated_data)
    # exit()

    # Train using the SGD:
    # SGD(5, templated_data)


if __name__ == '__main__':
    main(sys.argv[1:])
