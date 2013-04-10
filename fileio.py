""" fileio.py: File manager
"""
import random
import numpy as np
import aifc
from matplotlib import mlab
import cv2
import pylab as pl


def ReadAIFF(file):
	""" Read AIFF and convert to numpy array

		Args:
			file: string
				file to read

		Returns:
			numpy array containing whale audio clip
	"""
	s = aifc.open(file,'r')
	nFrames = s.getnframes()
	strSig = s.readframes(nFrames)
	return np.fromstring(strSig, np.short).byteswap()
	
class TrainData(object):
	""" Training data

		Store the files in two lists: h0 and h1
		h0 are the non-right whale clips
		h1 are the right whale clips

		Args:
			fileName: string
				csv file containing file names and labels
			dataDir: string
				directory the data lives in
	"""
	def __init__(self, fileName='', dataDir=''):
		""""""
		self.fileName = fileName
		self.dataDir = dataDir
		self.h1 = []
		self.h0 = []
		self.Load()

	def Load(self):
		""" Read the csv file and populate lists """
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
		
	def H1Sample(self, index=None, params=None):
		""" Grab an H1 sample

			Args:
				index: index of file to read
				params: dictionary containing specgram params

			Returns:
				Spectrogram and freq/time bins
		"""
		if index == None:
			index = random.randint(0,self.numH1-1)
			print index
		s = ReadAIFF(self.dataDir+self.h1[index])
		P, freqs, bins = mlab.specgram(s, **params)
		return P, freqs, bins
			
	def H0Sample(self, index=None, params=None):
		""" Grab an H0 sample

			Args:
				index: index of file to read
				params: dictionary containing specgram params

			Returns:
				Spectrogram and freq/time bins
		"""
		if index == None:
			index = random.randint(0,self.numH0-1)
		s = ReadAIFF(self.dataDir+self.h0[index])
		P, freqs, bins = mlab.specgram(s, **params)
		return P, freqs, bins
		
class TestData(object):
	""" Test data file manager

		Args:
			dataDir: string
				directory data lives in

	"""
	def __init__(self, dataDir=''):
		""""""
		self.dataDir = dataDir
		self.test = range(1,54504)
		self.nTest = 54503

	def TestSample(self, index=None, params=None):
		""" Grab an H0 sample

			Args:
				index: index of file to read
				params: dictionary containing specgram params

			Returns:
				Spectrogram and freq/time bins
		"""		
		if index == None:
			index = random.randint(1,self.nTest)
		s = ReadAIFF(self.dataDir+'test'+('%i'%index)+'.aiff')
		P, freqs, bins = mlab.specgram(s, **params)
		return P, freqs, bins

