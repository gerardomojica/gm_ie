# Test file to train a log-linear model: logistic regresssion
# 01/18/2016

import numpy as np
from scipy.optimize import minimize, fmin, fmin_bfgs

# Make the logistic regression/sigmoid function:
# f(x)=1/1+exp(-theta.T*x)

# Linear regression objective function
def obj_function(theta,x,y):
	return 0.5*(np.dot(x,theta)-y)

def sigmoid(z):
	return 1/(1+np.exp(z))

def sigmod_prime(w,x):
	return (w-w**2)*sigmoid(np.dot(x,w.T))
	return (w-w**2)*sigmoid(np.dot(x,w.T))

def test_sigmoid():
	w = np.array([[2,3,10],[2,3,10]])
	w = np.array([[2],[3],[10]])
	w = np.array([[2,3,10]])
	x = np.array([[0.3,0.2,0.5],[0.4,0.2,0.5]])

	
	print np.dot(x,w.T) 
	# print np.dot(w,x.T) 
	# print np.dot(w,x) # These two do the same, when x and y are just vector
	# print np.multiply(w,x.T) # this is elementwise multiplication!
	
	# Also, w*x is the same, elementwise
	print sigmoid(np.dot(x,w.T))
	print sigmod_prime(w,x)



	x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
	res = minimize(rosen, x0, method='nelder-mead',
	               options={'xtol': 1e-8, 'disp': True})


def test_linear_regression():
	theta = np.ones((2,1))
	x = np.random.rand(5,2)
	y = np.random.rand(5,1)

	# print theta
	# print x
	# print np.dot(x,theta)

	print obj_function(theta,x,y)

	# res = fmin(obj_function, theta, args=(x,y), xtol=1e-8, disp=True)
	res = fmin_bfgs(obj_function, theta, args=(x,y))


if __name__ == '__main__':
	# test_sigmoid()
	test_linear_regression()
	# test2 = lambda x: -(x[0]-x[0]**2 + x[1] - x[1]**2 )
	# print test2(np.array([[.5, .5],[3,2]]))
	# guess = [ 0.5,0.5 ] #just some guess
	# print fmin( test2, guess )






