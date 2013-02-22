
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
	else:
		H0 = np.loadtxt(baseDir+'workspace/h0mean.csv',delimiter=',')
		H1 = np.loadtxt(baseDir+'workspace/h1mean.csv',delimiter=',')
	
	# make a mask
	mask = filters.CreateMask(H1, 0.002)
	mask = filters.Dilate(mask)
	mask[15:,:] = 0
	P, freqs, bins = train.H1Sample(params=params)
	x, y, shifts = metrics.calcShifts(mask*H1, freqs, bins)
	plotting.PlotSpecLine(H1, freqs, bins, x, y)
	u = metrics.ShiftIntegrate(P,x,y,shifts)
	pl.figure()
	pl.plot(u)
	pl.show()
	plotting.PlotSpecgram(P, freqs, bins)

if __name__ == "__main__":
	main()