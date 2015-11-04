#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author Name: George Lee
# Email: georgelee01@sohu.com

#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: George Lee
# E-mail: georgelee01@sohu.com
#
#
import numpy as np

def readDataFromFile(path):
	f =  open(path,'r')
	lines = f.read().split('\r\n')

	paras = lines[0].split(' ')
	numSamples = int(paras[0])
	numInputs  = int(paras[1])
	numOutputs = int(paras[2])
	dataSet = np.zeros((numSamples, numInputs+numOutputs)) 
	lines = lines [1:]
	for index in xrange(numSamples):
		lineArr = lines[index].split('\t')
		for i in xrange(numInputs + numOutputs):
			dataSet[index,i] = float(lineArr[i])
	data_x = dataSet[:,0:numInputs]
	data_y = dataSet[:,numInputs:numInputs+numOutputs]	

	return data_x, data_y

if __name__ == '__main__':
	data_x, data_y = readDataFromFile('train-b.txt')
	print data_x, data_y
