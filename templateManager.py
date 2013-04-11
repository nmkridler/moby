""" templateManager.py
	Object that manages the templates
"""
import numpy as np
import pylab as pl
from matplotlib import mlab
import cv2
import fileio
import metrics

class TemplateManager(object):
	""" TemplateManager

		Chips out templates from files designated in a 
		user-defined input file.

		Args:
			filename: string
				csv file containing chip file and bounds
			trainObj: TrainData object
				access to the training data
			params: dict
				parameters for the Spectrogram

	"""
	def __init__(self, fileName='', trainObj=None, params=None):
		""""""
		self.train = trainObj
		self.params = params
		self.info = []
		self.file = open(fileName,'r')
		self.LoadFile()
		self.file.close()

		self.templates = []
		self.limits = []
		self.getTemplates()
		self.size = len(self.templates)

	def LoadFile(self):
		"""Load template file
				File should be organized as: fileNumber, xStart, xEnd, yStart, yEnd, class
				fileNumber is the index into the TrainData object
		"""
		self.file.readline()
		for line in self.file.readlines():
			tokens = line.split('\n')[0].split(',')
			fileMap = {'file': int(tokens[0]),'x0': int(tokens[1]), 'xn': int(tokens[2]),
						'y0': int(tokens[3]), 'yn': int(tokens[4]), 'class':tokens[5] }
			self.info.append(fileMap)
	
	def getTemplates(self):
		""" Chip out the templates """
		for maps in self.info:
			# Determine if it's an H1 or an H0 template
			if maps['class'] == 'H1':
				P, freqs, bins = self.train.H1Sample(maps['file'],params=self.params)
			else:
				P, freqs, bins = self.train.H0Sample(maps['file'],params=self.params)

			# Get the image and demean it
			m, n = P.shape
			P = metrics.slidingWindowV(P,maxM=m)
			x0, xn, y0, yn = maps['x0'], maps['xn'], maps['y0'], maps['yn']
			tmpl = P[x0:xn,y0:yn].astype('Float32')

			# Normalize the image and turn into a binary mask
			m_, s_ = np.mean(tmpl), np.std(tmpl)
			min_ = tmpl.min()
			tmpl[ tmpl < m_ + 0.5*s_] = min_ 
			tmpl[ tmpl > min_ ] = 1
			tmpl[tmpl < 0] = 0

			# Add meta-data to the list
			self.templates.append(tmpl)
			m, n = tmpl.shape
			self.limits.append([bins[y0], bins[yn], freqs[x0], freqs[xn]])

	def PlotTemplates(self, index=None):
		""""""
		if index == None:
			index = random.randint(0,len(self.templates))
		Z = np.flipud(self.templates[index])
		
		x0, xn, y0, yn = self.limits[index]
		extent = x0, xn, y0, yn
		pl.figure()
		im = pl.imshow(Z, extent=extent)
		pl.axis('auto')
		pl.xlim([x0, xn])
		pl.ylim([y0, yn])
		pl.show()		
