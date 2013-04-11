""" genTrainMetrics.py

	This file generates the training metrics
"""


import numpy as np
import pylab as pl

import metrics
import plotting
import fileio
import templateManager
import cv2

def main():
	baseDir = '/home/nick/whale/' # Base directory

	###################### SET OUTPUT FILE NAME HERE ########################
	trainOutFile = baseDir+'workspace/trainMetrics.csv'

	
	############################## PARAMETERS ###############################
	dataDir = baseDir+'data/'				   # Data directory
	params = {'NFFT':256, 'Fs':2000, 'noverlap':192} # Spectogram parameters
	maxTime = 60 # Number of time slice metrics

	######################## BUILD A TrainData OBJECT #######################
	train = fileio.TrainData(dataDir+'train.csv',dataDir+'train/')

	##################### BUILD A TemplateManager OBJECT ####################
	tmplFile = baseDir+'moby/templateReduced.csv'
	tmpl = templateManager.TemplateManager(fileName=tmplFile, 
		trainObj=train, params=params)

	################## VERTICAL BARS FOR HIFREQ METRICS #####################
	bar_ = np.zeros((12,9),dtype='Float32')
	bar1_ = np.zeros((12,12),dtype='Float32')
	bar2_ = np.zeros((12,6),dtype='Float32')
	bar_[:,3:6] = 1.
	bar1_[:,4:8] = 1.
	bar2_[:,2:4] = 1.

	########################### CREATE THE HEADER ###########################
	outHdr = metrics.buildHeader(tmpl)

	###################### LOOP THROUGH THE FILES ###########################
	hL = []
	for i in range(train.numH1):
		P, freqs, bins = train.H1Sample(i,params=params)
		out = metrics.computeMetrics(P, tmpl, bins, maxTime)
		out += metrics.highFreqTemplate(P, bar_)
		out += metrics.highFreqTemplate(P, bar1_)
		out += metrics.highFreqTemplate(P, bar2_)
		hL.append([1, i] + out)
	for i in range(train.numH0):
		P, freqs, bins = train.H0Sample(i,params=params)
		out = metrics.computeMetrics(P, tmpl, bins, maxTime)
		out += metrics.highFreqTemplate(P, bar_)
		out += metrics.highFreqTemplate(P, bar1_)
		out += metrics.highFreqTemplate(P, bar2_)
		hL.append([0, i] + out)
	hL = np.array(hL)
	file = open(trainOutFile,'w')
	file.write("Truth,Index,"+outHdr+"\n")
	np.savetxt(file,hL,delimiter=',')
	file.close()
		

if __name__ == "__main__":
	main()
