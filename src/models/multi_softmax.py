# Luis Mojica - June. 2015
# Information Extraction - Opinion Mining - Graphical Model
# The University of Texas at Dallas
# Contains the script calls

import sys
import numpy as nu
from SoftMax import SoftMax
from sklearn.feature_extraction.text import FeatureHasher
import re

# h = FeatureHasher(n_features=10,input_type='string',non_negative=True)

def tokens(doc):
    """Extract tokens from doc.

    This uses a simple regex to break strings into tokens. For a more
    principled approach, see CountVectorizer or TfidfVectorizer.
    """
    return (tok.lower() for tok in re.findall(r"\w+", doc))

# Create observation features qs:
def obs_featurizer(single_observation):
	unigrams = list(tokens(single_observation))

	return unigrams

# Return a vector of features:
def featurize_y_x(hasher,y_clique_string,X_string=None):

	f_strings = []
	if X_string:
		for x_str in X_string:
			f_strings.append([y_clique_string+'_#'+x_str])
	else:
		f_strings.append([y_clique_string])

	return hasher.transform(f_strings)

# This is a softmax function p(y|x;theta)=ø(w,y,x):
def unary_function(theta,features):
	return sm.softmaxDist(self,theta,features)

def main(args):
	n_features = 20

	# Create a weights vector, assinging a weight to each feature:
	# theta = 


	# Create a feature hasher:
	hasher = FeatureHasher(n_features=n_features,input_type='string',non_negative=True)

	obs_features = obs_featurizer('This is not trolling')
	f_vec = featurize_y_x(hasher,'I0',obs_features)

	


	# Create a uniary factor:

	

if __name__ == '__main__':
	main(sys.argv[1:])



