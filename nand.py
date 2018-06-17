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
optFound = False
runs = 80000
#split into segments to test cost vs benefit of runs
#choose n within 2 ordes of magnitude of 'runs'
num_segments = int(runs/100)
rate = 3 #this is the % between segments that will make a segement the 'optimum'

for i in range(runs):
	#layer 1 is the input layer X
	l1 = X
	#layer 2 is X multiplied by the randomly generated weight matrix, w1
	l2 = nonlin(np.dot(l1, w1))
	#layer 3 is layer 2 mulitplied by random weight matrix w2
	#layer 3 is also the output layer
	l3 = nonlin(np.dot(l2, w2))

	Er_l3 = y - l3
	Er_list.append(np.mean(abs(Er_l3)))

	#aka error of output mulitplied by the derivative of the output
	#this find how far off our weights are
	l3_delta = Er_l3*nonlin(l3, deriv=True)

	Er_l2 = l3_delta.dot(w2.T)

	l2_delta = Er_l2*nonlin(l2, deriv=True)

	#keep within 1 order of magnitude of runs
	if(i % 10000 == 0):
		print('Error at run #{}: '.format(i), np.mean(np.abs(Er_l3)))
		#print(w2,'\n')
	

	#update weights
	w2 += l2.T.dot(l3_delta)
	w1 += l1.T.dot(l2_delta)

	if(len(Er_list) > (num_segments*2) and (optFound == False)):
		#if the relative difference between segents is less
		# than 'rate' then we've found a (relatively) optimal number of runs
		if(100*np.abs((Er_list[i-num_segments]- Er_list[i])/Er_list[i-num_segments]) < rate):
			optPoint = i
			optFound = True


#print('Input: ',X)
#print('Output: ',l3)
#print('Actual: ',y)

#print('\n\n\n',Er_list[2000] - Er_list[4000])
fig, ax = plt.subplots()


#ax.annotate('figure pixels',
            #xy=(optPoint, Er_list[optPoint]), xycoords='figure pixels')


if(optFound):
	print('Optimum stopping point at', optPoint)
	ax.annotate('optimum',
    	xy=(optPoint, Er_list[optPoint]), xycoords='data',
        xytext=(-15, 25), textcoords='offset points',
        arrowprops=dict(facecolor='black', shrink=0.05),
        horizontalalignment='right', verticalalignment='bottom')



plt.plot(Er_list)
plt.show()



















