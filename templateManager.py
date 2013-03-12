import numpy as np
import pylab as pl
from matplotlib import mlab

import fileio
import filters
import metrics
reload(fileio)
reload(filters)
reload(metrics)

class TemplateManager(object):
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
		self.m = []
		self.n = []
		self.getTemplates()
		self.size = len(self.templates)

	def LoadFile(self):
		"""Load template file
				File should be organized as: fileNumber, xStart, xEnd, yStart, yEnd
		"""
		self.file.readline()
		for line in self.file.readlines():
			tokens = line.split('\n')[0].split(',')
			fileMap = {'file': int(tokens[0]),'x0': int(tokens[1]), 'xn': int(tokens[2]),
								 'y0': int(tokens[3]), 'yn': int(tokens[4]) }
			self.info.append(fileMap)
	
	def getTemplates(self):
		"""Read templates"""
		for maps in self.info:
			P, freqs, bins = self.train.H1Sample(maps['file'],params=self.params)
			m, n = P.shape
			P = metrics.slidingWindow(P, maxM=m)
			x0, xn, y0, yn = maps['x0'], maps['xn'], maps['y0'], maps['yn']
			tmpl = P[x0:xn,y0:yn].astype('Float32')
			self.templates.append(tmpl)
			m, n = tmpl.shape
			self.m.append(m)
			self.n.append(n)
			self.limits.append([bins[y0], bins[yn], freqs[x0], freqs[xn]])

	def PlotTemplates(self, index=None):
		"""Spectrogram"""
		if index == None:
			index = random.randint(0,len(self.templates))
		#Z = 10. * np.log10(self.templates[index])
		Z = np.flipud(self.templates[index])
		
		x0, xn, y0, yn = self.limits[index]
		extent = x0, xn, y0, yn
		pl.figure()
		im = pl.imshow(Z, extent=extent)
		pl.axis('auto')
		pl.xlim([x0, xn])
		pl.ylim([y0, yn])
		pl.show()		
