
import numpy as np
import pylab as pl

import metrics
import plotting
import filters
import fileio

reload(plotting)
reload(metrics)
reload(filters)
reload(fileio)

def main():
	# Parameters
	outputAverage = False
	baseDir = 'C:/Users/Nick/Desktop/RightWhale/'
	dataDir = baseDir+'data/whale_data/data/'
	params = {'NFFT':64, 'Fs':2000, 'noverlap':48}

	# Training Data
	train = fileio.TrainData(dataDir+'train.csv',dataDir+'train/')
	
	# Output averages
	if outputAverage:
		h1name = baseDir+'workspace/h1mean.csv'
		h0name = baseDir+'workspace/h0mean.csv'
		fileio.OutputAverages(train, h0name, h1name, params)
		
	P, freqs, bins = train.H1Sample(params=params)
	plotting.PlotSpecgram(P, freqs, bins)
	P, freqs, bins = train.H0Sample(params=params)
	plotting.PlotSpecgram(P, freqs, bins)

if __name__ == "__main__":
	main()