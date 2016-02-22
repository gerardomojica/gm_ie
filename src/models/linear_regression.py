# Test file to train a linear model: linear regresssion
# 01/25/2016

import numpy as np
from scipy.optimize import minimize, fmin, fmin_bfgs
import scipy.optimize as op
from SoftMax import SoftMax
# from numpy.random import randn,randint

# Linear model
def linear_model(w,x):
	return np.dot(w.T,x)

def gradient(y,w,x):
	# Elementwise multiplication, then sum by collumns
	return np.sum(x*(y-linear_model(w,x)),1)

# The cost function for the linear regression:
def cost_function(y,w,x):
	# Elementwise multiplication, then sum by collumns
	return np.sum(x*(y-linear_model(w,x)),1)


def update(w,grad,alpha):
	w = w.T+(alpha*grad)
	return w.T

def gradient_descent(grad,alpha,w,x,y,it):
	w_prev=w
	for idx in xrange(0,it):
		grad = gradient(y,w,x)
		w = update(w,grad,alpha)

		# Early finish:
		if np.allclose(w,w_prev,atol=1e-08) and idx>0:
		# if np.array_equal(w,w_prev) and idx>0:
			print idx
			return w
		w_prev = w
	return w

def stochastic_gradient_descent(grad,alpha,w,x,y,it):
	w_prev=w
	for idx in xrange(0,it):
		# Here is the difference, use one training instance a the time:
		for col in xrange(x.shape[1]):
			x_inst = x[:,col].reshape(x.shape[0],1)
			y_inst = y[:,col].reshape(y.shape[0],1)
			# print x_inst
			# print y_inst
			grad = gradient(y_inst,w,x_inst)
			w = update(w,grad,alpha)
			# print w

			# Early finish:
			if np.allclose(w,w_prev,atol=1e-08) and idx>0:
			# if np.array_equal(w,w_prev) and idx>0:
				print idx
				return w
			w_prev = w
	return w


def test_linear_model():


	# An example:
	# This is row vector
	# f(x)= y = b+w_1*x = w_0*1+w_1*x_1
	y = np.array([[0.5,1,1.5,2,2.5]])
	x = np.array([[1,1,1,1,1],[-1,0,1,2,3]])
	w = np.array([[0],[0]])

	# print linear_model(w,x)
	grad = gradient(y,w,x)
	# w = update(w,grad,1)
	# print grad
	# print w
	
	print gradient_descent(grad,0.1,w,x,y,1000)
	# print stochastic_gradient_descent(grad,0.1,w,x,y,100)

def test_lbfgs():

	# An example:
	# This is row vector
	# f(x)= y = b+w_1*x = w_0*1+w_1*x_1
	y = np.array([[0.5,1,1.5,2,2.5]])
	x = np.array([[1,1,1,1,1],[-1,0,1,2,3]])
	w = np.array([[0],[0]])

	# print linear_model(w,x)
	grad = gradient(y,w,x)

	x = op.fmin_l_bfgs_b(CostFunc, x0=initial_theta, fprime=Gradient, args= (X, y))
	print x

#####################################
#####################################
#####################################
# Logistic regression:
class LogisticRegression(object):
	"""docstring for LogisticRegression"""
	def __init__(self):
		super(LogisticRegression, self).__init__()

	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	def log_linear_model(self,w,x):
		# print self.sigmoid(np.dot(w.T,x))
		return self.sigmoid(np.dot(w.T,x))

	def gradient(self,y,w,x):
		# print (y-self.log_linear_model(w,x))*x
		return np.sum((y-self.log_linear_model(w,x))*x,1)

	def update(self,w,grad_val,alpha):
		w = w.T+(alpha*grad_val)
		return w.T

	def binary_accuracy(self,y,w,x):
		correct = np.sum(y==(self.log_linear_model(w,x)>0.5))
		return float(correct)/len(y[0])


	def batch_gradient_descent(self,alpha,w,x,y,it):
		w_prev=w
		for idx in xrange(0,it):
			grad_val = self.gradient(y,w,x)
			# print w
			w = self.update(w,grad_val,alpha)

			# Early finish:
			if np.allclose(w,w_prev,atol=1e-08) and idx>0:
			# if np.array_equal(w,w_prev) and idx>0:
				print idx
				return w
			w_prev = w

		return w

	# These two are to be used with a lbfgs:
	def cost_function(self,w,x,y):
		# return -lg.log_linear_model(w,x)
		ln_mod = self.log_linear_model(w,x)
		return -np.sum(y*np.log(ln_mod)+((1-y)*np.log(1-ln_mod)))

	def opt_gradient(self,w,x,y):
		return -np.sum((y-self.log_linear_model(w,x))*x,1)



def test_logistic_regression():

	# Single parameter:
	y = np.array([[1,1,0,0,1]])
	# x = np.array([[-4,-2,1,3]])
	# Using a binary feature:
	x = np.array([[1,1,1,1,1],[1,1,0,0,0]])
	w = np.array([[-1],[-1]])
	# x = np.array([[1,1,0,0]])
	# w = np.array([[1]])
	# print y,x
	# print np.dot(w.T,x)
	# exit()
	
	lg = LogisticRegression()
	# print lg.sigmoid(0)
	# print lg.log_linear_model(w,x)
	# # exit()
	# print y-lg.log_linear_model(w,x)
	# grad_val = np.sum((y-lg.log_linear_model(w,x))*x,1)
	# print 1+grad_val*0.01
	# w = w + 0.01*grad_val
	# print w
	# print lg.gradient(y,w,x)
	# exit()
	# print lg.binary_accuracy(y,w,x)
	w = lg.batch_gradient_descent(0.01,w,x,y,1000)
	print w
	print lg.binary_accuracy(y,w,x)

def test_logitic_regession_optimize():

	# Single parameter:
	y = np.array([[1,1,0,0,1]])
	# x = np.array([[-4,-2,1,3]])
	# Using a binary feature:
	x = np.array([[1,1,1,1,1],[1,1,0,0,1]])
	w = np.array([[0],[0]])
	# x = np.array([[1,1,0,0,1]])
	# w = np.array([[0]])

	lg = LogisticRegression()

	cost_function = lg.cost_function(w,x,y)
	gradient_fun = lg.opt_gradient(w,x,y)

	# x = op.fmin_l_bfgs_b(cost_function, x0=w, fprime=gradient_fun, args=(y,x))
	# w = op.fmin_bfgs(lg.cost_function, x0=w, fprime=lg.opt_gradient, args=(x,y))
	w = op.fmin_l_bfgs_b(lg.cost_function, w, fprime=lg.opt_gradient, args=(x,y))
	print w

def test_logitic_regession_optimize2():

	# Single parameter:
	# y = np.array([[0,0,0,0,0,1,1,1,1,1]])
	y = np.array([[1,1,1,1,1,0,0,0,0,0]])
	x = np.array([[1,1,1,1,1,1,1,1,1,1],[29,30,31,31,32,29,30,31,32,33],[62,83,74,88,68,41,44,21,50,33]])
	w = np.array([[6],[-2],[1]])

	# x = np.array([[29,30,31,31,32,29,30,31,32,33]])
	# w = np.array([[-2]])

	lg = LogisticRegression()

	print lg.binary_accuracy(y,w,x)
	# exit()

	cost_function = lg.cost_function
	gradient_fun = lg.opt_gradient
	# print sum(gradient_fun(w, *(x,y)))
	# print np.sqrt(sum(gradient_fun(w, *(x,y))))
	# print np.sqrt(sum(gradient_fun(w, *(x,y))))
	inc = [1.49e-04]
	print gradient_fun(w, *(x,y))
	print op.approx_fprime(w, cost_function, inc, *(x,y))
	print sum((gradient_fun(w, *(x,y))-op.approx_fprime(w, cost_function, inc, *(x,y)))**2)
	print np.sqrt(sum((gradient_fun(w, *(x,y))-op.approx_fprime(w, cost_function, inc, *(x,y)))**2))
	# print np.sqrt(sum(gradient_fun(w, *(x,y))-op.approx_fprime(w, cost_function, 1.49e-08, *(x,y))))
	exit()
	# return sqrt(sum((grad(x0, *args) - approx_fprime(x0, func, _epsilon, *args))**2))

	op.check_grad(gradient_fun, cost_function, w, (w,x,y))
	exit()

	res = op.fmin_l_bfgs_b(lg.cost_function, w, fprime=lg.opt_gradient, args=(x,y))
	w = res[0]
	print lg.binary_accuracy(y,w,x)
	# w

def test_values():
	
	y = np.array([[1,1,1,1,1,0,0,0,0,0]])
	x = np.array([[1,1,1,1,1,1,1,1,1,1],[29,30,31,31,32,29,30,31,32,33],[62,83,74,88,68,41,44,21,50,33]])
	w = np.array([[6],[-2],[1]])

	lg = LogisticRegression()
	print lg.log_linear_model(w,x)
	print lg.binary_accuracy(y,w,x)


def gradient_checker(f,g,w,args):
	
	print f(w,*args)
	print g(w,*args)

def test_gradiente_checker():
	lg = LogisticRegression()

	# A logitic regression problem:
	y = np.array([[1,1,1,1,1,0,0,0,0,0]])
	x = np.array([[1,1,1,1,1,1,1,1,1,1],[29,30,31,31,32,29,30,31,32,33],[62,83,74,88,68,41,44,21,50,33]])
	w = np.array([[6],[-2],[1]])

	gradient_checker(lg.cost_function,lg.opt_gradient,w,(x,y))

class SoftMaxClassifier(object):
	"""docstring for SoftMaxClassifier"""
	def __init__(self):
		super(SoftMaxClassifier, self).__init__()
		# self.arg = arg

	def softmax(self,w,x):
		print np.dot(w.T,x)/(1)

def test_softmax():

	y = np.array([[1,1,1,1,1,0,0,0,0,0]])
	x = np.array([[1,1,1,1,1,1,1,1,1,1],[29,30,31,31,32,29,30,31,32,33],[62,83,74,88,68,41,44,21,50,33]])
	w = np.array([[6],[-2],[1]])
	print w.T
	exit()


	sm = SoftMaxClassifier()

	sm.softmax(w,x)

def test_softmax_module():
	
	sm = SoftMax()

	# Data
	# labels = np.array([[1,1,1,1,1,0,0,0,0,0,2,2]])
	# inputData = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],[29,30,31,31,32,29,30,31,32,33,100,99],[62,83,74,88,68,41,44,21,50,33,3,4]])

	labels = np.array([[1,1,1,1,1,0,0,0,0,0,]])
	inputData = np.array([[1,1,1,1,1,1,1,1,1,1],[29,30,31,31,32,29,30,31,32,33],[62,83,74,88,68,41,44,21,50,33]])

	inputSize = inputData.shape[0] # dimensionality
	numClasses = len(np.unique(labels))
	eta = 1e-4

	## Randomly initialize theta
	# theta = 0.005 * np.random.randn(numClasses * inputSize,1)
	# theta = np.zeros(numClasses * inputSize)
	theta = np.zeros((numClasses, inputSize))

	## Find gradient and current softmax cost
	cost, grad = sm.softmaxCost(theta, numClasses, inputSize, eta, inputData,
			labels)

	# Learn Parameters
	maxIter = 100
	cost, optTheta, d = sm.softmaxTrain(inputSize, numClasses, eta, inputData, labels,
			maxIter)

	# Test on True Dataset
	testData = inputData
	testLabels = labels

	pred = sm.softmaxPredict(optTheta, testData)
	dist = sm.softmaxDist(optTheta, testData)
	acc = np.mean(testLabels == pred)

	print dist
	print("Accuracy: " + str(acc * 100))


# An observation function that extracts features:
def obsFeatures(xtext,feat_list):
	
	import re

	pattern = re.compile(r'^[A-Z\d]+$')
	if 'all_caps' in feat_list:


def main():
	# test_linear_model()

	# test_logistic_regression()

	# test_lbfgs()

	# test_logitic_regession_optimize()

	# test_logitic_regession_optimize2()

	# test_gradiente_checker() # work in progress

	# test_values()

	# test_softmax() # Work in progress

	test_softmax_module() # This one works great!

if __name__ == '__main__':
	main()


