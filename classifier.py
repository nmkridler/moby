import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from plotting import *
from sklearn.metrics import roc_curve, auc

def getAuc(clf,X,y,n_estimators):
	#test_deviance = np.zeros((n_estimators,), dtype=np.float64)
	test_auc = np.zeros((n_estimators,), dtype=np.float64)
	for i, y_pred in enumerate(clf.staged_decision_function(X)):
		if i % 20 == 0:
		# clf.loss_ assumes that y_test[i] in {0, 1}
		#test_deviance[i] = clf.loss_(y, y_pred)
			fpr, tpr, thresholds = roc_curve(y,y_pred)
			test_auc[i] = auc(fpr,tpr)
	#pl.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],'-')
	pl.plot((np.arange(test_auc.shape[0]) + 1)[::20], test_auc[::20],'-',lw=2)


class Classify(object):
	def __init__(self, trainFile=''):
		self.loadTrain(trainFile)
		self.m, self.n = self.train.shape
		self.scaler = StandardScaler()
		self.scaler.fit(self.train)
		
	def loadTrain(self, trainFile=''):
		d_ = pd.read_csv(trainFile)
		self.truth = np.array(d_.Truth)
		self.index = np.array(d_.Index)
		self.train = np.array(d_.ix[:,2:])
		self.hdr = d_.columns[2:]
		m, n = self.train.shape
		#ignore_ = [ 12, 14, 17, 69, 74, 86, 247, 248] + range(239,246)
		#indices = [j for j in range(n) if j not in ignore_]
		#indices = range(140) + range(159,309) + range(359,n)  # + range(347,n) # 0.97659 -500
		#print [self.hdr[j] for j in indices]
		#self.train = self.train[:,np.array(indices)]
		#self.hdr = [self.hdr[i] for i in indices]
		if False:
			indices = []
			file = open("/Users/nkridler/Desktop/whale/workspace/features.csv",'r')
			for line in file.readlines()[:120]:
				indices.append(int(line.split('\n')[0]))
			file.close()
			self.indices = np.array(indices)
			self.train = self.train[:, np.array(indices)]

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

		#pl.figure()
		params = clf.get_params()	
		for fold in range(nFolds-1):
			print "Fold: %i" % fold
			trainIdx = (idx >= (fold+1)*delta) | (idx < fold*delta)
			testIdx = (idx >= fold*delta) & (idx < (fold+1)*delta)
			clf.fit(X[trainIdx,:],y[trainIdx])
			y_[testIdx,:] = clf.predict_proba(X[testIdx,:])
			getAuc(clf,X[testIdx,:],y[testIdx],params['n_estimators'])
	
		# Last fold	
		trainIdx = idx < (nFolds-1)*delta
		testIdx = idx >= (nFolds-1)*delta
		clf.fit(X[trainIdx,:],y[trainIdx])
		y_[testIdx,:] = clf.predict_proba(X[testIdx,:])
		getAuc(clf,X[testIdx,:],y[testIdx],params['n_estimators'])
		pl.ylim([0.96,0.985])
		pl.show()

		PlotROC(y, y_[:,1])
		pl.show()
		# Save to file
		outP = np.empty((self.m,3))
		for i in range(self.m):
			outP[i,:] = np.array([y[i], self.index[self.p[i]], y_[i,1]])
		np.savetxt("/Users/nkridler/Desktop/whale/workspace/probaBaseGBM.csv",outP,delimiter=',')

		if featureImportance:
			print "Feature ranking:"
			importances = clf.feature_importances_
			indices = np.argsort(importances)[::-1]
			for f in xrange(min(100,self.n)):
				print "%d. feature (%s,%f)" % (f + 1, self.hdr[indices[f]], importances[indices[f]])
				#print "%d. feature (%d,%f)" % (f + 1, indices[f], importances[indices[f]])
			if False:	
				out = open("/Users/nkridler/Desktop/whale/workspace/features.csv",'w')
				for f in xrange(self.n):
					#outStr="%d. feature (%s,%f)" % (f + 1, self.hdr[indices[f]], importances[indices[f]])
					out.write("%d\n"%indices[f])
					out.write(outStr+'\n')
				out.close()	
	
	def testAndOutput(self, clf=None, testFile='', outfile='sub.csv'):
		test = self.scaler.transform(np.loadtxt(testFile, delimiter=',',skiprows=1))
		#test = self.scaler.transform(np.loadtxt(testFile, delimiter=',')[:,self.indices])
		X, y = self.shuffle()
		clf.fit(X,y)
		y_ =  clf.predict_proba(test)
		np.savetxt(outfile,y_[:,1],delimiter=',')
