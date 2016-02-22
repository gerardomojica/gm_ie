# Test create the functions for opengm
# 01/25/2016

import numpy as np
from scipy.optimize import minimize, fmin, fmin_bfgs
import scipy.optimize as op
from SoftMax import SoftMax
import re
from sklearn.feature_extraction.text import FeatureHasher
from numpy.random import randn,randint
from sklearn.feature_extraction.text import CountVectorizer

class FeatFunctions(object):
	"""docstring for featFunctions"""
	def __init__(self,n_features=None):

		# self.arg = arg
		import re
		from sklearn.feature_extraction.text import FeatureHasher
		from numpy.random import randn,randint
		from sklearn.feature_extraction.text import CountVectorizer

		# Define some parameters:
		if not n_features:
			n_features = 100000

		# Initialize the hasher:
		self.hasher = FeatureHasher(n_features=n_features,input_type='string',non_negative=True)

		# Initialize the ngram:
		self.vectorizer = CountVectorizer(binary=True)

		# Feature name-function dictionary:
		self.featName_function={'url':self.url,'all_caps':self.all_caps,'ngrams':self.ngrams}

		
	def all_caps(self,x):
		pat = re.compile(r'^[A-Z\d]+$')
		groups = pat.match(x)
		if groups:
			return ['f_all_caps']

	def url(self,x):
		pat = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
		groups = pat.findall(x)
		if groups:
			return ['f_url']

	def ngrams(self,x):
		ngram_feats = self.vectorizer.fit_transform([x])
		return self.vectorizer.inverse_transform(ngram_feats)[0].tolist()

	# An observation function that extracts features. x is a raw text
	def getObsFeatures(self,x,feat_list):
		str_feats = []
		for feat in feat_list:
			feat = feat(x)
			if feat:
				str_feats+=feat

		return str_feats

	def getYXFeatures(self,y_name,y_idx,obs_feat_list):
		# return y_name+'_'+str(y_idx).join(obs_feat_list)
		# return map(lambda x,y:x+y,y_name+'_'+str(y_idx),obs_feat_list)
		xy_feat = [y_name+str(y_idx)+'_'+xfeat for xfeat in obs_feat_list]
		# print xy_feat

		hashed_feats = self.hasher.transform([xy_feat])
		# return hashed_feats.nonzero()[1]
		return hashed_feats

def all_caps(x):
		pat = re.compile(r'^[A-Z\d]+$')
		groups = pat.match(x)
		if groups:
			return ['f_all_caps']

def url(x):
	pat = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	groups = pat.findall(x)
	if groups:
		return ['f_url']

def ngrams(x):
	ngram_feats = vectorizer.fit_transform([x])
	return vectorizer.inverse_transform(ngram_feats)[0].tolist()

# An observation function that extracts features. x is a raw text
def getObsFeatures(x,feat_list):
	str_feats = []
	for feat in feat_list:
		feat = feat(x)
		if feat:
			str_feats+=feat

	return str_feats

def getYXFeatures(y_name,y_idx,obs_feat_list):
	# return y_name+'_'+str(y_idx).join(obs_feat_list)
	# return map(lambda x,y:x+y,y_name+'_'+str(y_idx),obs_feat_list)
	xy_feat = [y_name+str(y_idx)+'_'+xfeat for xfeat in obs_feat_list]
	# print xy_feat

	hashed_feats = hasher.transform([xy_feat])
	# return hashed_feats.nonzero()[1]
	return hashed_feats

def main():
	global vectorizer
	vectorizer = CountVectorizer(binary=True)

	n_features = 5000

	# Create a weights vector, assinging a weight to each feature:
	theta = 0.005 * randn(n_features)

	# Create a feature hasher:
	global hasher
	hasher = FeatureHasher(n_features=n_features,input_type='string',non_negative=True)

	feat_list = [url,all_caps,ngrams]
	# obs_feat_list = getObsFeatures('JONAS http://www.reddit.com, esto esta de pelos',feat_list)
	obs_feat_list = getObsFeatures('esto esta de pelos',feat_list)
	
	featureIdx = getYXFeatures('I',0,obs_feat_list)
	featureIdx = getYXFeatures('I',1,obs_feat_list)
	featureIdx = getYXFeatures('I',2,obs_feat_list)

	featureIdx = getYXFeatures('R',0,obs_feat_list)
	featureIdx = getYXFeatures('R',1,obs_feat_list)


	print featureIdx.dot(theta)[0]


if __name__ == '__main__':
	main()


