#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author Name: George Lee
# Email: georgelee01@sohu.com

import numpy as np
import dataLoad
import svm
import svmutil
import os


class BCM(object):
	def __init__(self,inputDim,outputDim):
		self.inputDim = inputDim
		self.outputDim = outputDim
		self.models = []

	def saveModels(self, Dir):
		print 'Saving models...'
		if not os.path.isdir(Dir):
			os.mkdir(Dir)
		for index in xrange(self.inputDim):
			filename = Dir + '/model_' + str(index)
			svmutil.svm_save_model(filename, self.models[index])
		print 'Saving models...Done.'

	def readModels(self, Dir):
		print 'Reading models...'
		if not os.path.isdir(Dir):
			print '[E]Cannot find out models, Please check the given directory!'
			return
		self.models = []
		for index in xrange(self.inputDim):
			filename = Dir + '/model_' + str(index)
			if not os.path.isfile(filename):
				print '[E]Wrong models!'
				self.models = []
				return
			model = svmutil.svm_load_model(filename)
			self.models.append(model)
		print 'Reading models...Done.'

	def modelTrain(self, trainFile):
		print 'Training models...'
		data_x, data_y = dataLoad.readDataFromFile(trainFile)
		#self.inputDim = data_x.shape[1]
		#self.outputDim = data_y.shape[1]
		if self.inputDim != data_x.shape[1]:
			print "[E]Input training data error!"
			return

		if self.outputDim != data_y.shape[1]:
			print "[E]Output training data error!"
			return

		for index in xrange(self.outputDim):
			model = svmutil.svm_train(data_y[:,index].tolist(),data_x.tolist(),'-s 4 -t 2 -q')
			self.models.append(model)
		print "Training models...Done."


	def predict(self, test_x):
		print "Predicting..."
		if self.inputDim != test_x.shape[1]:
			print '[E]Input testset error!'
			return None
		numSamples = test_x.shape[0]
		test_y = np.zeros((numSamples, self.outputDim))
		vec_sum = []
		for i in xrange(numSamples):
			vec_sum.append([])
		for index in xrange(self.outputDim):
			if numSamples == 1:
				a,b,c = svmutil.svm_predict([test_y.tolist()],[test_x.tolist()],self.models[index],'-q')
			else:
				a,b,c = svmutil.svm_predict(test_y.tolist(),test_x.tolist(),self.models[index],'-q')
			vec_sum = np.hstack((vec_sum,c))
		print 'Predicting...Done.'
		return vec_sum


#need to def scale and trans-scale functions