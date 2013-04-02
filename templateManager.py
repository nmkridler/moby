import numpy as np
import pylab as pl
from matplotlib import mlab
import cv2
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
		self.averages = []
		self.counts = []
		self.limits = []
		self.m = []
		self.n = []
		self.flag = []
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
						'y0': int(tokens[3]), 'yn': int(tokens[4]), 'class':tokens[5] }
			self.info.append(fileMap)
	
	def getTemplates(self):
		"""Read templates"""
		e_ = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
		for maps in self.info:
			# Determine if it's an H1 or an H0 template
			self.flag.append(maps['class'])
			if maps['class'] == 'H1':
				P, freqs, bins = self.train.H1Sample(maps['file'],params=self.params)
			else:
				P, freqs, bins = self.train.H0Sample(maps['file'],params=self.params)

			# Get the image and demean it
			m, n = P.shape
			P = metrics.slidingWindowV(P,maxM=m)
			x0, xn, y0, yn = maps['x0'], maps['xn'], maps['y0'], maps['yn']
			tmpl = P[x0:xn,y0:yn].astype('Float32')

			# Normalize the image and blur it
			m_, s_ = np.mean(tmpl), np.std(tmpl)
			min_ = tmpl.min()
			tmpl[ tmpl < m_ + 0.5*s_] = min_ 
			tmpl[ tmpl > min_ ] = 1
			tmpl[tmpl < 0] = 0
			#tmpl = cv2.erode(tmpl,e_,iterations=1) 
			#tmpl = cv2.dilate(tmpl,e_,iterations=1) 
			#tmpl = cv2.GaussianBlur(tmpl,(3,3),0)

			# Add meta-data to the list
			self.templates.append(tmpl)
			self.averages.append(np.zeros(tmpl.shape))
			self.counts.append(0)
			m, n = tmpl.shape
			self.m.append(m)
			self.n.append(n)
			self.limits.append([bins[y0], bins[yn], freqs[x0], freqs[xn]])

	def PlotAverages(self, index=0):
		""""""
		Z = np.flipud(self.averages[index])
		x0, xn, y0, yn = self.limits[index]
		extent = x0, xn, y0, yn
		pl.figure()
		im = pl.imshow(Z, extent=extent)
		pl.axis('auto')
		pl.xlim([x0, xn])
		pl.ylim([y0, yn])
		pl.show()	

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
