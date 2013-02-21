import random
import numpy as np
import aifc
import scipy.fftpack as spFFT
from matplotlib import mlab

def ReadAIFF(file):
	s = aifc.open(file,'r')
	nFrames = s.getnframes()
	strSig = s.readframes(nFrames)
	return np.fromstring(strSig, np.short).byteswap()

def OutputAverages(train, h0name='', h1name='', params):
	avgP, freqs, bins = train.H1Sample(0)
	for index in range(1,train.numH1):
		P, freqs, bins = train.H1Sample(index)
		avgP += P
	np.savetxt(h1name, avgP/train.numH1, delimiter=',')
	
	avgP, freqs, bins = train.H0Sample(0)
	for index in range(1,train.numH0):
		P, freqs, bins = train.H0Sample(index)
		avgP += P
	np.savetxt(h0name, avgP/train.numH0, delimiter=',')	
	
class TrainData(object):
	def __init__(self, fileName='', dataDir=''):
		self.fileName = fileName
		self.dataDir = dataDir
		self.h1 = []
		self.h0 = []
		self.Load()

	def Load(self):
		file = open(self.fileName, 'r')
		self.hdr = file.readline().split('\n')[0].split(',')
		
		for line in file.readlines():
			tokens = line.split('\n')[0].split(',')
			if int(tokens[1]) == 0:
				self.h0.append(tokens[0])
			else:
				self.h1.append(tokens[0])
		file.close()
		self.numH1 = len(self.h1)
		self.numH0 = len(self.h0)
		
	def H1Sample(self, index=None, params):
		if index == None:
			index = random.randint(0,self.numH1-1)
		s = ReadAIFF(self.dataDir+self.h1[index])
		s = (s - np.mean(s))/np.std(s)
		return mlab.specgram(s, **params)
			
	def H0Sample(self, index=None, params):
		if index == None:
			index = random.randint(0,self.numH0-1)
		s = ReadAIFF(self.dataDir+self.h0[index])
		s = (s - np.mean(s))/np.std(s)
		return mlab.specgram(s, **params)
		
class TrainData(object):
	def __init__(self, fileName='', dataDir=''):
		self.fileName = fileName
		self.dataDir = dataDir
		self.test = range(1,54504)
		self.nTest = 54503

	def TestSample(self, index=None, params):
		if index == None:
			index = random.randint(1,self.nTest)
		s = ReadAIFF(dataDir+'test'+('%i'%index)+'.aiff')
		s = (s - np.mean(s))/np.std(s)
		return mlab.specgram(s, **params)