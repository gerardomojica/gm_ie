#!/bin/bash
# Luis Mojica - Feb. 2016
# The University of Texas at Dallas
# Create the model for test and train the graphical model for
# joint trolling detection:

import sys

# Create a class that import the other needed classes:
class BuildGM(object):
	"""docstring for BuildGM"""
	def __init__(self,n_features=None):
		from featFunctions import FeatFunctions
		import numpy as np
		import opengm
		from numpy.random import randn,randint
		
		# Initialize the FeatureFunction object:
		self.FF = FeatFunctions(n_features)

	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	# Create functions for factors:
	def funCreator(self,name,y_name_list,feat_list_names,obsString):
		# print name
		
		# Get the set of functions using the dicionary:
		feature_set = []
		for feature_name in feat_list_names:
			if feature_name in self.FF.featName_function:
				feature_set.append(self.FF.featName_function[feature_name])
			else:
				print feature_name,'not found'

		# Get the list of observed features:
		obs_feat_list = self.FF.getObsFeatures(obsString,feature_set)

		# The function to return
		def foo(y_idxs):
				
			# This is used when the function is defined over more than one variable:
			y_name_list_str = '_'.join(y_name_list)
			y_idx_str = '_'.join(str(x) for x in y_idxs)

			# Get the features conditioned to y_clique
			featureIdx = FF.getYXFeatures(y_name_list_str,y_idx_str,obs_feat_list)
			# print featureIdx,'\n'
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
	s0 = 'esto esta chido'

	# Define the feature set for this factor:
	# feat_list_names_factor0 = ['url','all_caps','ngrams']
	feat_list_names_factor0 = ['ngrams']

	# Define the functions for the factors:
	# phi0 = funCreator('phi0','I',feat_list_names_factor0,s0,FF)

	phi0 = funCreator('phi0',['I'],feat_list_names_factor0,s0,FF)
		
	# Initalize a graphical model:
	# For this example, one single variable, logistic regression
	numVars = 1
	cardinality = [3] # Binary classifier
	
	# Create the gm:
	gm=opengm.gm(cardinality*numVars,'multiplier')

	# Transform the function to opengm:
	# The second parameter is a list of the cardinalities of the variables in the scope of this function
	py_func_phi0 = opengm.PythonFunction(phi0,[3])
    
 	# Add the function the pgm
	gm_func_phi0 = gm.addFunction(py_func_phi0)

	# # Add the opengm function to the gm model
	# The second parameter is the [set] of variables in the scope of this factor
	gm.addFactor(gm_func_phi0,0)

	# Run inference to get the variable marginals:
	bp = opengm.inference.BeliefPropagation(gm, 'integrator')
	bp.infer()
	
	# Get a list of all variables in the sentence:
	var_idx = [x for x in xrange(gm.numberOfVariables)]
	marg = bp.marginals(var_idx)
	print 'marg:\n',marg

# Load the dataset:
def load_annotated_threads(filename=None):
	
	# Artificial Data:
	dataset = []
	instance = {'sP':["Parent's sentence"],'sT':["Troll's sentence"],'sC':["Commenter 0 sentence","Commenter 1 sentence"]}

	# Add the instance to the dataset:
	dataset.append(instance)

	return dataset

def groundPGM(PMG,instance,factors_feature_set,factors_obs_set,vars_to_facs):
	
	# # Observed text:
	# obs_tex_

	# Variables dictionary:
	vars_dic = {'I0':0,'D0':1}
	var_idx = 2

	# Go over all the comments:
	for comIdx,commentator in enumerate(instance['sC']):
		vars_dic['R'+str(comIdx)]=var_idx
		vars_dic['B'+str(comIdx)]=var_idx+1

		# Increase the counter:
		var_idx+=2

	# Create a relation of variables indices and their fucntions:
	variable_functions = {}

	# Create the unary pgm functions:
	for factor_name, factor_features in factors_feature_set.iteritems():

		# Get the sentence to extract the observation features for this factor:
		if factor_name in factors_obs_set:

			# Get the factor type
			stype = factors_obs_set[factor_name]
			# print factor_name,stype
			
			for s in stype:
				# A factor type may have multiple occurrences:
				for senIdx, sentence in enumerate(instance[s]):
					# print senIdx, sentence

					var_name = '_'.join(vars_to_facs[factor_name])+str(senIdx)
					print factor_name,'_'.join(vars_to_facs[factor_name])+str(senIdx)
					
					# Get the variable index:
					if var_name in vars_dic:
						var_idx = vars_dic[var_name]
						
						# Make the function:
						variable_functions[var_idx]=PMG.funCreator(factor_name,[var_name],factor_features,instance[s])
						exit()

					else:
						print var_name,'something is wrong, variable not find in variable dictionary'

	print variable_functions

def main():

	# Define the number or features:
	n_features = 5000

	# Load dataset: (for now, artificial instances)
	dataset = load_annotated_threads(filename=None)
	instance = dataset[0]

	# Define feature sets for each of the factors:
	factors_feature_set={'phi0':['ngrams'],'phi1':['ngrams'],'phi2':['ngrams'],'phi3':['ngrams'],'phi4':['ngrams'],'phi5':['ngrams'],'phi6':['ngrams'],'phi7':['ngrams']}

	# Define the observations give to each of the factors:
	factors_obs_set={'phi0':['sT'],'phi1':['sP'],'phi2':['sC'],'phi6':['sC'],'phi7':['sP']}

	# Define the observations give to each of the factors:
	vars_to_facs={'phi0':['I'],'phi1':['D'],'phi2':['B'],'phi3':['D','R'],'phi4':['I','R'],'phi5':['R,','B'],'phi6':['R'],'phi7':['I']}


	# Initialize the pgm builder:
	PMG = BuildGM(n_features=5000)

	# Init ngrams:
	PMG = BuildGM(n_features=5000)

	# Ground a pgm for the current fucntion:
	groundPGM(PMG,instance,factors_feature_set,factors_obs_set,vars_to_facs)

if __name__ == '__main__':
	main()


