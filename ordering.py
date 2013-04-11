""" ordering.py: generate ordering feature
"""

import numpy as np
import pylab as pl
import pandas as pd
import fileio


def writeToFile(out,outname='out.csv'):
	""""""
	file = open(outname,'w')
	np.savetxt(file,out,delimiter=',')
	file.close()

def orderMetric(truth,n):
	"""Calculate the average excluding the center"""
	pad_ = np.zeros(2*n+truth.size)
	pad_[n:truth.size+n] = truth.copy()
	return np.array([(np.sum(pad_[(i-n):i+n]) - pad_[i])/(2*n-1) for i in range(n,pad_.size-n)])

def main():
	baseDir = '/home/nick/whale/'
	dataDir = baseDir+'data/'

	# Open up the train file
	train = fileio.TrainData(dataDir+'train.csv',dataDir+'train/')
	t_ = pd.read_csv(dataDir+'train.csv')
	order32_ = orderMetric(t_.label,32)
	order64_ = orderMetric(t_.label,64)

	# Reorder the data
	reorder32 = order32_.copy()
	reorder64 = order64_.copy()
	k = 0
	for i in xrange(train.numH1):
		j = int(train.h1[i].split('.')[0][5:]) - 1
		reorder32[k] = order32_[j] 
		reorder64[k] = order64_[j] 
		k += 1
	for i in xrange(train.numH0):
		j = int(train.h0[i].split('.')[0][5:]) - 1
		reorder32[k] = order32_[j] 
		reorder64[k] = order64_[j] 
		k += 1

	writeToFile(reorder32,'corr32.csv')
	writeToFile(reorder64,'corr64.csv')

	# There are 84503 samples
	trainSize = 30000
	testSize = 54503
	size_ = trainSize + testSize

	tt_ = size_*2. # since these are 2 second clips
	xs_ = np.linspace(0,tt_,testSize)
	xt_ = np.linspace(0,tt_,trainSize)
	test32_ = np.interp(xs_,xt_,order32_)
	test64_ = np.interp(xs_,xt_,order64_)

	writeToFile(test32_,'testCorr32.csv')
	writeToFile(test64_,'testCorr64.csv')

if __name__=="__main__":
	main()