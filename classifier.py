import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from plotting import *
from sklearn.metrics import roc_curve, auc

def getAuc(clf,X,y,n_estimators):
	""" Calculate the auc at different iterations """
	test_auc = np.zeros((n_estimators,), dtype=np.float64)
	for i, y_pred in enumerate(clf.staged_decision_function(X)):
		if i % 20 == 0:
			fpr, tpr, thresholds = roc_curve(y,y_pred)
			test_auc[i] = auc(fpr,tpr)
	pl.plot((np.arange(test_auc.shape[0]) + 1)[::20], test_auc[::20],'-',lw=2)


class Classify(object):
	""" Classification Object
	"""
	def __init__(self, trainFile='', orderFile=None, useCols=None):
		""""""
		self.trainFile = trainFile
		self.orderFile = orderFile
		self.indices = useCols
		self.loadTrain()
		self.m, self.n = self.train.shape
		self.scaler = StandardScaler()
		self.scaler.fit(self.train)
		
	def loadTrain(self):
		""" Load the training data

			Args:
				trainFile: string
					training metrics file
		"""
		d_ = pd.read_csv(self.trainFile)
		if self.orderFile is not None:
			d_['proba'] = np.loadtxt(self.orderFile,delimiter=',')

		# Set truth, index, train, and header
		self.truth = np.array(d_.Truth)
		self.index = np.array(d_.Index)
		self.train = np.array(d_.ix[:,2:])
		self.hdr = d_.columns[2:]
	
		if self.indices is not None:
			self.train = self.train[:,self.indices]
			self.hdr = [self.hdr[i] for i in self.indices]
		else:
			self.indices = np.arange(self.hdr.size)	

	def shuffle(self, seed=0):
		""" Shuffle and scale

			Args:
				seed: random seed

			Returns:
				shuffled and scaled data
		"""
		self.p = range(self.m)
		random.seed(seed)
		random.shuffle(self.p)
		return self.scaler.transform(self.train[self.p,:]), self.truth[self.p]
		
	def validate(self, clf, nFolds=4, featureImportance=False, plotAuc=False, outFile=None):
		""" Validate
		"""
		X, y = self.shuffle()
		
		# Figure out big the folds are
		delta = self.m/nFolds
		idx = np.arange(self.m)
		y_ = np.empty((self.m,2))

		if plotAuc:
			pl.figure()
		params = clf.get_params()	
		for fold in range(nFolds-1):
			print "Fold: %i" % fold
			train_ = (idx >= (fold+1)*delta) | (idx < fold*delta)
			test_ = (idx >= fold*delta) & (idx < (fold+1)*delta)
			clf.fit(X[train_,:],y[train_])
			y_[test_,:] = clf.predict_proba(X[test_,:])
			if plotAuc:
				getAuc(clf,X[test_,:],y[test_],params['n_estimators'])
	
		# Last fold	
		train_ = idx < (nFolds-1)*delta
		test_ = idx >= (nFolds-1)*delta
		clf.fit(X[train_,:],y[train_])
		y_[test_,:] = clf.predict_proba(X[test_,:])

		if plotAuc:
			getAuc(clf,X[test_,:],y[test_],params['n_estimators'])
			pl.ylim([0.96,0.985])
			pl.show()

		# Plot the roc curve
		PlotROC(y, y_[:,1])
		pl.show()

		# Save to file
		if outFile is not None:
			outP = np.empty((self.m,3))
			for i in range(self.m):
				outP[i,:] = np.array([y[i], self.index[self.p[i]], y_[i,1]])
			np.savetxt(outFile,outP,delimiter=',')

		# Print top 100 Features
		if featureImportance:
			print "Feature ranking:"
			importances = clf.feature_importances_
			indices = np.argsort(importances)[::-1]
			for f in xrange(min(100,self.n)):
				print "%d. feature (%s,%f)" % (f + 1, self.hdr[indices[f]], importances[indices[f]])
	
	def testAndOutput(self, clf=None, testFile='', orderFile=None, outfile='sub.csv'):
		""" Build a classifier and test it 

			Args:
				clf: classifier object
				testFile: test metrics file
				outFile: submission file

		"""
		tf_ = pd.read_csv(testFile)

		if orderFile is not None:
			tf_['proba'] = np.loadtxt(orderFile,delimiter=',')
		
		test = self.scaler.transform(np.array(tf_)[:,self.indices])
		X, y = self.shuffle()
		clf.fit(X,y)
		y_ =  clf.predict_proba(test)
		np.savetxt(outfile,y_[:,1],delimiter=',')
