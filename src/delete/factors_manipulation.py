import opengm
import numpy as np
# print dir(opengm)
# print opengm.Adder.__dict__
# exit()
# #------------------------------------------------------------------------------------
# # This example shows how multiple  unaries functions and functions / factors add once
# #------------------------------------------------------------------------------------
# # add unaries from a for a 2d grid / image
# width=2
# height=2
# numVar=width*height
# numLabels=2
# # construct gm
# gm=opengm.gm(np.ones(numVar,dtype=opengm.label_type)*numLabels)
# # construct an array with all numeries (random in this example)
# unaries=np.random.rand(width,height,numLabels)
# # reshape unaries is such way, that the first axis is for the different functions
# unaries2d=unaries.reshape([numVar,numLabels])
# # add all unary functions at once (#numVar unaries)
# fids=gm.addFunctions(unaries2d)
# # np array with the variable indices for all factors
# vis=np.arange(0,numVar,dtype=np.uint64)
# # add all unary factors at once
# gm.addFactors(fids,vis)

# # Print factors:
# # print list(fids)
# # print list(gm.factors())
# # print gm[0],gm[1]
# # print gm[0]+gm[1]
# a = gm[0]+gm[1]
# b = a+gm[1]
# # print b

# opengm.Adder.op(gm[1],a)
# Explicit function:
fu1 = np.array([[2,3],[4,5]])
fu2 = np.array([[6,7],[8,9]])
# fu1 = np.array([2,3,4,5])
# fu2 = np.array([6,7,8,9])



gm2 = opengm.gm([2,2,2],'multiplier')
f1=gm2.addFunction(fu1)
f2=gm2.addFunction(fu2)

gm2.addFactor(f1,[0,1])
gm2.addFactor(f2,[1,2])
fac0 = gm2[0]
fac1 = gm2[1]

joint = fac0*fac1
summation = fac0+fac1
# print dir(summation)

print gm2.evaluate([0,0,0])
print gm2.evaluate([1,1,1])

# for x in joint:
# 	print x



