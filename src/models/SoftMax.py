from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from numpy.random import randn,randint
from numpy.linalg import norm

class SoftMax(object):
  """docstring for SoftMax"""
  def __init__(self):
    super(SoftMax, self).__init__()
    
    # Code to perform softmax
    self.EPSILON = 1e-4

  def computeNumericalGradient(self,J, theta):
    """ theta: parameter vector
        J: function that outputs a real number (i.e., y = J(theta))
    """
    numGrad = np.zeros(theta.shape)
    for i in range(len(theta)):

      # Compute the thetas without copying the vector
      theta[i] = theta[i] + self.EPSILON
      JthetaP = J(theta)
      theta[i] = theta[i] - 2 * self.EPSILON
      JthetaN = J(theta)
      theta[i] = theta[i] + self.EPSILON

      numGrad[i] = (JthetaP - JthetaN)/(2 * self.EPSILON)
    return numGrad

  def softmaxPredict(self,theta, data):
    """ data - the input matrix
        produces preduction matrix pred, where pred(i) = argmax_c P(y(c) | x(i))
    """
    costMatrix = np.exp(np.dot(theta, data))
    # columnSums = sum(costMatrix, axis=0)
    columnSums = sum(costMatrix)
    probMatrix = costMatrix / columnSums

    pred = np.argmax(probMatrix, axis=0)
    return pred

  def softmaxDist(self,theta,data):
    """ data - the input matrix
        produces preduction matrix pred, where pred(i) = argmax_c P(y(c) | x(i))
    """
    # cM = np.dot(theta, data)
    # cM = cM - np.amax(cM, axis=0)
    # cM = np.exp(cM)
    # # cS = sum(cM, axis=0)
    # cS = sum(cM)
    # cM = cM / cS
    # lCM = np.log(cM)

    costMatrix = np.exp(np.dot(theta, data))
    columnSums = sum(costMatrix)
    probMatrix = costMatrix / columnSums

    return probMatrix

  def softmaxFactor(self,y_str_lst,theta,qx):
    """ data - the input matrix
        produces preduction matrix pred, where pred(i) = argmax_c P(y(c) | x(i))
    """
    # cM = np.dot(theta, data)
    # cM = cM - np.amax(cM, axis=0)
    # cM = np.exp(cM)
    # # cS = sum(cM, axis=0)
    # cS = sum(cM)
    # cM = cM / cS
    # lCM = np.log(cM)

    costMatrix = np.exp(np.dot(theta, qx))
    columnSums = sum(costMatrix)
    probMatrix = costMatrix / columnSums

    return probMatrix

  def softmaxTrain(self,inputSize, numClasses, eta, inputData, labels, maxIter):
    theta = 0.005 * randn(numClasses * inputSize, 1)
    # Make the memory fortran contiguous
    theta = theta.copy('f')
    inputData = inputData.copy('f')
    labels = labels.copy('f')
    J = lambda x: self.softmaxCost(x, numClasses, inputSize, eta, inputData, labels)
    optTheta, cost, d = fmin_l_bfgs_b(J, theta, maxfun = maxIter)
    optTheta = np.reshape(optTheta, (numClasses, inputSize), order='F')
    return cost, optTheta, d

  def softmaxCost(self,theta, numClasses, inputSize, eta, data, labels, l2=True):
    """ numClasses - number of label classes
        inputSize - the size of input vector
        eta - weight decay parameter
        data - the input matrix whose column d[:,i] corresponds to
               a single test example
        labels - a vector containing labels corresponding to input data
    """
    theta = np.reshape(theta, (numClasses, inputSize), order='F')
    # numCases = shape(data)[1]
    numCases = data.shape[1]
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
    # thetaGrad = np.dot(groundTruth - cM, transpose(data))
    thetaGrad = np.dot(groundTruth - cM, data.T)

    # # This thing contains a regularizer already!
    if l2:
      cost = -cost / numCases + eta * norm(theta, 'fro')**2 / 2
      thetaGrad = -thetaGrad / numCases + eta * theta
    else:
      # Without regularizer:
      cost = -cost / numCases
      thetaGrad = -thetaGrad

    grad = thetaGrad.flatten('F')
    return cost, grad

def test_gradient():

  # Instantiate the class:
  sm = SoftMax()
  
  # Initialize Parameters
  inputSize = 28 * 28
  numClasses = 10
  eta = 1e-4

  # If debugging, reduce the input size
  DEBUG = True
  if DEBUG:
      numClasses = 2 # k
      inputSize = 3 # d
      numCases = 5 # n
      inputData = randn(inputSize,numCases)
      #inputData = ones((inputSize, numCases))
      labels = randint(numClasses,size=(numCases,1))
      #labels = vint(ones((numCases,1)))
      labels[0] = 0



  ## Randomly initialize theta
  theta = 0.005 * randn(numClasses * inputSize,1)

  print inputData
  print theta
  exit()


  ## Find gradient and current softmax cost
  cost, grad = sm.softmaxCost(theta, numClasses, inputSize, eta, inputData,
          labels)

  ## If in debug mode, check the numerical gradient
  if DEBUG:
    Jsoftmax = lambda x: sm.softmaxCost(x, numClasses, inputSize,
        eta, inputData, labels)[0]
    numGrad = sm.computeNumericalGradient(Jsoftmax, theta)
    numGrad = np.reshape(numGrad, (numClasses, inputSize), order='F')
    numGrad = numGrad.flatten('F')

    # Compute the numeric gradient to the analytic gradient
    diff = norm(numGrad - grad,2)/norm(numGrad + grad,2)
    print str(diff)
    print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)'
    #thetaP = theta.copy()
    #thetaN = theta.copy()

def test_mnist():
  # Needs the image files and their importer
  # from loadmnist import *
  # Initialize Parameters

  # Instantiate the class:
  sm = SoftMax()

  inputSize = 28 * 28
  numClasses = 10
  eta = 1e-4

  ## Load the Data
  images =  loadMNISTImages('train-images-idx3-ubyte')
  labels =  loadMNISTLabels('train-labels-idx1-ubyte')
  inputData = images

  ## Randomly initialize theta
  theta = 0.005 * randn(numClasses * inputSize,1)

  ## Find gradient and current softmax cost
  cost, grad = sm.softmaxCost(theta, numClasses, inputSize, eta, inputData,
          labels)

  # Learn Parameters
  maxIter = 100
  cost, optTheta, d = sm.softmaxTrain(inputSize, numClasses, eta, inputData, labels,
          maxIter)

  # Test on True Dataset
  testData = loadMNISTImages('t10k-images-idx3-ubyte')
  testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte')

  pred = sm.softmaxPredict(optTheta, testData)
  acc = np.mean(testLabels == pred)

  print("Accuracy: " + str(acc * 100))

def test_distribution():
  # Instantiate the class:
  sm = SoftMax()

  numClasses = 2 # k
  inputSize = 3 # dimensionality
  numCases = 5 # n
  inputData = randn(inputSize,numCases)
  #inputData = ones((inputSize, numCases))
  labels = randint(numClasses,size=(numCases,1))
  #labels = vint(ones((numCases,1)))
  labels[0] = 0

  ## Randomly initialize theta
  theta = 0.005 * randn(numClasses,inputSize)

  # Test the distribution:
  # print sm.softmaxDist(theta, inputData)

  # Another way of doing this:
  y_str_lst=['0','1']
  theta = np.reshape(theta,(-1,1))
  qx = np.reshape(inputData,(-1,1))
  # print theta.T
  # print qx
  print np.dot(theta.T, qx)
  # print sm.softmaxFactor(y_str_lst,theta,qx)


if __name__ == "__main__":
  # print 'this is a module with the SoftMax class in it'
  # test_gradient()  
  # test_mnist()
  test_distribution()
