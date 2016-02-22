# from pylab import *
from loadmnist import *
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from numpy.random import randn,randint
from numpy.linalg import norm

# Code to perform softmax
EPSILON = 1e-4

def computeNumericalGradient(J, theta):
  """ theta: parameter vector
      J: function that outputs a real number (i.e., y = J(theta))
  """
  numGrad = np.zeros(shape(theta))
  for i in range(len(theta)):

    # Compute the thetas without copying the vector
    theta[i] = theta[i] + EPSILON
    JthetaP = J(theta)
    theta[i] = theta[i] - 2 * EPSILON
    JthetaN = J(theta)
    theta[i] = theta[i] + EPSILON

    numGrad[i] = (JthetaP - JthetaN)/(2 * EPSILON)
  return numGrad

def softmaxPredict(theta, data):
  """ data - the input matrix
      produces preduction matrix pred, where pred(i) = argmax_c P(y(c) | x(i))
  """
  costMatrix = np.exp(np.dot(theta, data))
  # columnSums = sum(costMatrix, axis=0)
  columnSums = sum(costMatrix)
  probMatrix = costMatrix / columnSums

  pred = np.argmax(probMatrix, axis=0)
  return pred

def softmaxTrain(inputSize, numClasses, eta, inputData, labels, maxIter):
  theta = 0.005 * randn(numClasses * inputSize, 1)
  # Make the memory fortran contiguous
  theta = theta.copy('f')
  inputData = inputData.copy('f')
  labels = labels.copy('f')
  J = lambda x: softmaxCost(x, numClasses, inputSize, eta, inputData, labels)
  optTheta, cost, d = fmin_l_bfgs_b(J, theta, maxfun = maxIter)
  optTheta = reshape(optTheta, (numClasses, inputSize), order='F')
  return cost, optTheta, d

def softmaxCost(theta, numClasses, inputSize, eta, data, labels):
  """ numClasses - number of label classes
      inputSize - the size of input vector
      eta - weight decay parameter
      data - the input matrix whose column d[:,i] corresponds to
             a single test example
      labels - a vector containing labels corresponding to input data
  """
  theta = reshape(theta, (numClasses, inputSize), order='F')
  numCases = shape(data)[1]
  groundTruth = np.zeros((numClasses, numCases))
  groundTruth[labels.flatten(),np.arange(numCases)] = 1
  thetaGrad = np.zeros((numClasses, inputSize))

  cM = np.dot(theta, data)
  cM = cM - np.amax(cM, axis=0)
  cM = np.exp(cM)
  # cS = sum(cM, axis=0)
  cS = sum(cM)
  cM = cM / cS
  lCM = np.log(cM)
  cost = sum(lCM[groundTruth==1])
  thetaGrad = np.dot(groundTruth - cM, transpose(data))

  cost = -cost / numCases + eta * norm(theta, 'fro')**2 / 2
  thetaGrad = -thetaGrad / numCases + eta * theta

  grad = thetaGrad.flatten('F')
  return cost, grad

if __name__ == "__main__":

  # vint = vectorize(int)
  # Initialize Parameters
  inputSize = 28 * 28
  numClasses = 10
  eta = 1e-4

  ## Load the Data
  images =  loadMNISTImages('train-images-idx3-ubyte')
  labels =  loadMNISTLabels('train-labels-idx1-ubyte')
  inputData = images

  # If debugging, reduce the input size
  DEBUG = False
  if DEBUG:
      numClasses = 10 # k
      inputSize = 100 # d
      numCases = 1000 # n
      inputData = randn(inputSize,numCases)
      #inputData = ones((inputSize, numCases))
      labels = randint(numClasses,size=(numCases,1))
      #labels = vint(ones((numCases,1)))
      labels[0] = 0

  ## Randomly initialize theta
  theta = 0.005 * randn(numClasses * inputSize,1)

  ## Find gradient and current softmax cost
  cost, grad = softmaxCost(theta, numClasses, inputSize, eta, inputData,
          labels)

  ## If in debug mode, check the numerical gradient
  if DEBUG:
    Jsoftmax = lambda x: softmaxCost(x, numClasses, inputSize,
        eta, inputData, labels)[0]
    numGrad = computeNumericalGradient(Jsoftmax, theta)
    numGrad = reshape(numGrad, (numClasses, inputSize), order='F')
    numGrad = numGrad.flatten('F')

    # Compute the numeric gradient to the analytic gradient
    diff = norm(numGrad - grad,2)/norm(numGrad + grad,2)
    print str(diff)
    print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)'
    #thetaP = theta.copy()
    #thetaN = theta.copy()

  else:
    # Learn Parameters
    maxIter = 100
    cost, optTheta, d = softmaxTrain(inputSize, numClasses, eta, inputData, labels,
            maxIter)

    # Test on True Dataset
    testData = loadMNISTImages('t10k-images-idx3-ubyte')
    testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte')

    pred = softmaxPredict(optTheta, testData)
    acc = np.mean(testLabels == pred)

    print("Accuracy: " + str(acc * 100))
