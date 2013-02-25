import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from plotting import *

class Classify(object):
	def __init__(self, trainFile=''):
		self.train = []
		self.truth = []
		self.loadTrain(trainFile)
		self.m, self.n = self.train.shape
		self.scaler = StandardScaler()
		self.scaler.fit(self.train)
		
	def loadTrain(self, trainFile=''):
		file = open(trainFile,'r')
		self.hdr = file.readline().split('\n')[0].split(',')[1:]
		for line in file.readlines():
			tokens = line.split('\n')[0].split(',')
			self.truth.append(float(tokens[0]))
			self.train.append([float(x) for x in tokens[1:]])
		self.train = np.array(self.train)
		self.truth = np.array(self.truth)
		file.close()
		
	def shuffle(self, seed=0):
		self.p = range(self.m)
		random.seed(seed)
		random.shuffle(self.p)
		return self.scaler.transform(self.train[self.p,:]), self.truth[self.p]
		
	def validate(self, clf, nFolds=4, featureImportance=False):
		X, y = self.shuffle()
		
		delta = self.m/nFolds
		idx = np.arange(self.m)
		y_ = np.empty((self.m,2))
		for fold in range(nFolds-1):
			print "Fold: %i" % fold
			trainIdx = (idx >= (fold+1)*delta) | (idx < fold*delta)
			testIdx = (idx >= fold*delta) & (idx < (fold+1)*delta)
			clf.fit(X[trainIdx,:],y[trainIdx])
			y_[testIdx,:] = clf.predict_proba(X[testIdx,:])
	
		# Last fold	
		trainIdx = idx < (nFolds-1)*delta
		testIdx = idx >= (nFolds-1)*delta
		clf.fit(X[trainIdx,:],y[trainIdx])
		y_[testIdx,:] = clf.predict_proba(X[testIdx,:])
		
		PlotROC(y, y_[:,1])
		# Save to file
		outP = np.empty((self.m,3))
		for i in range(self.m):
			outP[i,:] = np.array([y[i], self.p[i], y_[i,1]])
		np.savetxt("/Users/nkridler/Desktop/whale/workspace/proba.csv",outP,delimiter=',')

		if featureImportance:
			print "Feature ranking:"
			importances = clf.feature_importances_
			indices = np.argsort(importances)[::-1]
			for f in xrange(min(10,self.n)):
				print "%d. feature (%d,%f)" % (f + 1, indices[f], importances[indices[f]])
		
		
	def testAndOutput(self, clf=None, testFile='', outfile='sub.csv'):
		test = self.scaler.transform(np.loadtxt(testFile, delimiter=','))
		X, y = self.shuffle()
		clf.fit(X,y)
		y_ =  clf.predict_proba(test)
		np.savetxt(outfile,y_[:,1],delimiter=',')
