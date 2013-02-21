import random
import numpy as np

import cv2
import cv2.cv as cv
from scipy import polyval, polyfit

def generateMetrics(files, functions):
	metrics = []
	for file in files:
		s = ReadAIFF(files)
		s = (s - np.mean(s))/np.std(s)
		m = []
		for f in functions:
			m += f(s)
		metrics += s
	return metrics

def lineFit(P, freqs, bins):
	m, n = P.shape
	wt = np.array([ P[:,j].argmax() for j in range(n)])
	v = np.array([ P[:,j].max() for j in range(n)])
	y = [ w for w in wt if w > 0]
	x = [ i for i in range(n) if wt[i] > 0]
	weight = [ v[i] for i in range(n) if wt[i] > 0 ]
	z = np.polyfit(x,y,1,w=weight)
	return [z[0], z[1]]

def matchTemplate(P, template):
	mf = cv2.matchTemplate(P.astype('Float32'), 
						   template, cv2.TM_CCOEFF_NORMED)
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mf)
	return [mf.max(), maxLoc[0] ,maxLoc[1]]

def correlate(x, y):
	xx, yy, xy = np.dot(x,x), np.dot(y,y), np.dot(x,y)
	return [xy/(np.sqrt(xx*yy))]

def calcShifts(P, freqs, bins):
	m, n = P.shape
	wt = np.array([ P[:,j].argmax() for j in range(n)])
	y = [ w for w in wt if w > 0]
	x = [ i for i in range(n) if wt[i] > 0]	
	shifts = [ y[i] - y[0] for i in range(len(y))] # anchor at y[0]
	return x, y, shifts	

def ShiftIntegrate(P, x, y, shifts):
	for i in range(len(x)):
		P[:,x[i]] = np.roll(P[:,x[i]],-shifts[i])
	return np.sum(P,1)