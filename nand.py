#neural network for "nand" binary operator
import numpy as np
import matplotlib.pyplot as plt

#sigmoid function
def nonlin(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	else: 
		return 1/(1+np.exp(-x))

#input data
X = np.array([[0,0],
	[0,1],
	[1,0],
	[1,1]])

#correct output data
y = np.array([[0],
	[1],
	[1],
	[0]])

#assign seed so we always start with the same data
np.random.seed(1)

#generate random weights
w1 = 2*np.random.random((2,4)) - 1
w2 = 2*np.random.random((4,1)) - 1

#print('\nWeights 1: \n', w1,'\n\n')
#print('Weights 2:\n', w2,'\n')

#optimize weights

Er_list = []

for i in range(50000):
	#layer 1 is the input layer X
	l1 = X
	#layer 2 is X multiplied by the randomly generated weight matrix, w1
	l2 = nonlin(np.dot(l1, w1))
	#layer 3 is layer 2 mulitplied by random weight matrix w2
	#layer 3 is also the output layer
	l3 = nonlin(np.dot(l2, w2))

	Er_l3 = y - l3
	Er_list.append(Er_l3)

	#aka error of output mulitplied by the derivative of the output
	#this find how far off our weights are
	l3_delta = Er_l3*nonlin(l3, deriv=True)

	Er_l2 = l3_delta.dot(w2.T)

	l2_delta = Er_l2*nonlin(l2, deriv=True)

	if(i % 10000 == 0):
		print('Error at run #{}: '.format(i), np.mean(np.abs(Er_l3)))
		#print(w2,'\n')

	#update weights
	w2 += l2.T.dot(l3_delta)
	w1 += l1.T.dot(l2_delta)

print('Input: ',X)
print('Output: ',l3)
print('Actual: ',y)
plt.plot(Er_list)
plt.show()
