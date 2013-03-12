
import numpy as np
import pylab as pl
import random
import templateManager
import metrics
import plotting
import filters
import fileio
import cv2
reload(plotting)
reload(metrics)
reload(filters)
reload(fileio)
reload(templateManager)

def main():
	# Parameters
	outputAverage = False
	baseDir = '/Users/nkridler/Desktop/whale/'
	dataDir = baseDir+'data/'
	params = {'NFFT':256, 'Fs':2000, 'noverlap':192}
	train = fileio.TrainData(dataDir+'train.csv',dataDir+'train/')

	# Open up the metrics file
	mets = np.loadtxt(baseDir+'workspace/templatesSlide.csv',delimiter=",",skiprows=1)
	
	tmplFile = baseDir+'moby/templateList.csv'
	tmpl = templateManager.TemplateManager(fileName=tmplFile, trainObj=train, params=params)
	if True:
		for i in range(tmpl.size):
			tmpl.PlotTemplates(i)
		return

	t, i_, p_ = np.loadtxt(baseDir+'workspace/proba.csv',delimiter=',',unpack=True)
	t, i_ = t.astype('int'), i_.astype('int')
	if True: # Plot probabilities
		plotting.PlotDensity(p_[t==1],"H1",minval=p_.min(),maxval=p_.max()) 
		plotting.PlotDensity(p_[t==0],"H0",minval=p_.min(),maxval=p_.max()) 
		pl.show()

	h1len = int(np.sum(t))
	z = [ -69.99342911,  248.6264494,  -122.45428956,  126.40882405]
	f = np.poly1d(z)
	x = np.linspace(0,2,256)

	numSample = 10
	if False:
		i = [ i_[j] - h1len for j in range(p_.size) if t[j] == 0 and p_[j] > 0.2 and p_[j] < 0.5]
		# Shuffle
		random.shuffle(i)
		print "H0"
		for ii in i[:numSample]:
			print ii, mets[ii + h1len, 1:10]
			P, freqs, bins = train.H0Sample(ii,params=params)
			Q = metrics.slidingWindow(P)
			plotting.PlotSpecLine(Q, freqs[:50], bins, x, f(x))
			plotting.PlotSpecgram(P, freqs, bins)

	if True:
		print "H1"

		i = [ i_[j] for j in range(p_.size) if t[j] == 1 and p_[j] < 0.4]
		#i = [ ii for ii in i if np.mean(mets[ii,1:10]) < 0.4]
		print len(i)
		#random.seed(0)
		random.shuffle(i)
		#P, freqs, bins = train.H1Sample(i[0],params=params)
		for ii in i[:numSample]:
			print ii, mets[ii,1:10]
			P, freqs, bins = train.H1Sample(ii,params=params)
			Q = metrics.slidingWindow(P)
			plotting.PlotSpecLine(Q, freqs[:50], bins, x, f(x))
			plotting.PlotSpecgram(P, freqs, bins)
		return

if __name__ == "__main__":
	main()
	

