#!/bin/bash
# Luis Mojica - Feb. 2016
# The University of Texas at Dallas
# Logistic Regression as a graphical model using opengm

import sys
import getopt
# This works in this way because the file is called as a module!
import numpy as np
import opengm
from featFunctions import FeatFunctions

import re
from sklearn.feature_extraction.text import FeatureHasher
from numpy.random import randn,randint
from sklearn.feature_extraction.text import CountVectorizer

# Go over each of the sentences and build the modle:
def MAP():
    for sent_id, sentence in enumerate(templated_data):

        # Build the graphical model:
        # gm = instantiate_sentence(templated_data[sent_id])
        gm = instantiate_sentence(sentence)

        # # Run inference:
        # # The MAP
        bp = opengm.inference.BeliefPropagation(gm, 'integrator')
        bp.infer()
        # Get a list of all variables in the sentence:
        var_idx = [x for x in xrange(gm.numberOfVariables)]
        marg = bp.marginals(var_idx)
        print 'marg:\n',marg

def sigmoid(z):
	return 1/(1+np.exp(-z))

# Create functions for factors:
def funCreator(name,y_name,feat_list_names,obsString,FF):
	# print name
	
	# Get the set of functions using the dicionary:
	feature_set = []
	for feature_name in feat_list_names:
		if feature_name in FF.featName_function:
			feature_set.append(FF.featName_function[feature_name])
		else:
			print feature_name,'not found'

	# Get the list of observed features:
	obs_feat_list = FF.getObsFeatures(obsString,feature_set)

	# The function to return
	def foo(y_idx):
		
		featureIdx = FF.getYXFeatures(y_name,y_idx,obs_feat_list)
		print featureIdx,'\n'
		z = featureIdx.dot(theta)[0]

		# Do the sigmoid:
		return sigmoid(z)

    # Give a name to the function
	foo.__name__ = name
	return foo

def test_logistic_as_gm():
	
	# Define the number or features:
	n_features = 5000
	
	# Initialize the FeatureFunction object:
	FF = FeatFunctions(n_features=5000)

	# Create a weights vector, assinging a weight to each feature:
	global theta
	theta = 0.5 * randn(n_features)

	# Get a sentence:
	s0 = 'esto esta de pelos'

	# Define the feature set for this factor:
	feat_list_names = ['url','all_caps','ngrams']

	# Define the functions for the factors:
	phi0 = funCreator('phi0','I',feat_list_names,s0,FF)
		
	# Initalize a graphical model:
	# For this example, one single variable, logistic regression
	numVars = 1
	cardinality = [3] # Binary classifier
	
	# Create the gm:
	gm=opengm.gm(cardinality*numVars,'multiplier')

	# Transform the function to opengm:
	py_func = opengm.PythonFunction(phi0,[3])
    
 	# Add the function the pgm
	gm_func = gm.addFunction(py_func)

	# # Add the opengm function to the gm model
	gm.addFactor(gm_func,0)

	bp = opengm.inference.BeliefPropagation(gm, 'integrator')
	bp.infer()
	
	# Get a list of all variables in the sentence:
	var_idx = [x for x in xrange(gm.numberOfVariables)]
	marg = bp.marginals(var_idx)
	print 'marg:\n',marg



def main():
	test_logistic_as_gm()



if __name__ == '__main__':
	main()


