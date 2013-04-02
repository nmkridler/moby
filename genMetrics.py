
import numpy as np
import pylab as pl

import metrics
import plotting
import filters
import fileio
import templateManager
import cv2
reload(plotting)
reload(metrics)
reload(filters)
reload(fileio)
reload(templateManager)

def main():
	# Parameters
	baseDir = '/Users/nkridler/Desktop/whale/'
	dataDir = baseDir+'data/'
	params = {'NFFT':256, 'Fs':2000, 'noverlap':192}
	train = fileio.TrainData(dataDir+'train.csv',dataDir+'train/')

	# Template Manager
	tmplFile = baseDir+'moby/templateReduced.csv'
	tmpl = templateManager.TemplateManager(fileName=tmplFile, 
		trainObj=train, params=params)

	# If you want to add a file to the template list
	addFile = False
	if addFile:
		P, freqs, bins = train.H1Sample(0,params=params)
		H1 = np.loadtxt(baseDir+'workspace/h1mean256.csv',delimiter=',')
		tH1 = H1[16:26,18:32].astype('Float32')
		m, n = tH1.shape
		tmpl.flag.append('H1')
		tmpl.templates.append(tH1)
		tmpl.averages.append(np.zeros(tH1.shape))
		tmpl.counts.append(0)
		tmpl.limits.append([bins[18], bins[32], freqs[16], freqs[26]])
		tmpl.m.append(m)
		tmpl.n.append(n)
		tmpl.size += 1	

	# Get a sample for the freqs/bins
	P, freqs, bins = train.H0Sample(6671,params=params)
	bar_ = np.zeros((12,9),dtype='Float32')
	bar_[:,3:6] = 1.
	bar1_ = np.zeros((12,12),dtype='Float32')
	bar1_[:,4:8] = 1.

	################################################################
	# Build the header
	#
	# This is dependent on the order that the metrics are generated
	#
	################################################################
	hdr_ = []
	prefix_ = ['max','xLoc','yLoc']
	for p_ in prefix_:
		for i in range(tmpl.size):
			hdr_.append(p_+'_%07d'%tmpl.info[i]['file'])
	for p_ in prefix_:
		for i in range(tmpl.size):
			hdr_.append(p_+'H_%07d'%tmpl.info[i]['file'])

	# Add spectrum metrics
	#for i in range(40,bins.size):
	#	hdr_ += ['centFreq_%04d'%i]
	#for i in range(bins.size):
	#	hdr_ += ['ratio_%04d'%i]

	# Add time metrics
	for i in range(50):
		hdr_ += ['centTime_%04d'%i]
	for i in range(50):
		hdr_ += ['bwTime_%04d'%i]

	# Add time metrics
	for i in range(50):
		hdr_ += ['centOops_%04d'%i]
	#for i in range(50):
	#	hdr_ += ['bwOops_%04d'%i]

	# Add high frequency metrics
	hdr_ += ['CentStd','AvgBwd','hfCent','hfBwd']
	# Add high freq template metrics
	hdr_ += ['hfMax','hfMax2']
	outHdr = ','.join(hdr_)

	# Flags to control metric generation
	trainOutFile = baseDir+'workspace/baseTrain7.csv'
	testOutFile = baseDir+'workspace/baseTest7.csv'
	trainData = True
	writeToFile = True
	up_ = metrics.bounds(bins,freqs)
	if trainData:
		hL = []
		for i in range(train.numH1):
			if i % 1000 == 0:
				print i
			P, freqs, bins = train.H1Sample(i,params=params)
			Q = metrics.slidingWindowV(P,inner=3,maxM=40)
			W = metrics.slidingWindowH(P,inner=3,outer=32,maxM=60) # best: 3,32,60
			out = metrics.templateMetrics(Q, tmpl)	
			out += metrics.templateMetrics(W, tmpl)	
			#out += metrics.spectrumMetrics(P,Q,freqs,up_)
			out += metrics.timeMetrics(P,bins)
			out += metrics.oopsMetrics(P,bins)
			out += metrics.highFreqMetrics(P,bins)
			out += metrics.highFreqTemplate(P, bar_, bins)
			out += metrics.highFreqTemplate(P, bar1_, bins)
			hL.append([1, i] + out)
		for i in range(train.numH0):
			if i % 1000 == 0:
				print i
			P, freqs, bins = train.H0Sample(i,params=params)
			Q = metrics.slidingWindowV(P,inner=3,maxM=40)
			W = metrics.slidingWindowH(P,inner=3,outer=32,maxM=60)
			out = metrics.templateMetrics(Q, tmpl)	
			out += metrics.templateMetrics(W, tmpl)	
			#out += metrics.spectrumMetrics(P,Q,freqs,up_)
			out += metrics.timeMetrics(P,bins)
			out += metrics.oopsMetrics(P,bins)
			out += metrics.highFreqMetrics(P,bins)
			out += metrics.highFreqTemplate(P, bar_, bins)
			out += metrics.highFreqTemplate(P, bar1_, bins)
			hL.append([0, i] + out)
		hL = np.array(hL)
		print hL.shape
		file = open(trainOutFile,'w')
		file.write("Truth,Index,"+outHdr+"\n")
		np.savetxt(file,hL,delimiter=',')
		file.close()


	else:
		test = fileio.TestData(dataDir+'test/')
		hL = []
		for i in range(test.nTest):
			if i % 1000 == 0 : 
				print i
			P, freqs, bins = test.TestSample(i+1,params=params)
			Q = metrics.slidingWindowV(P,inner=3,maxM=40)
			W = metrics.slidingWindowH(P,inner=3,outer=32,maxM=60)
			out = metrics.templateMetrics(Q, tmpl)	
			out += metrics.templateMetrics(W, tmpl)	
			#out += metrics.spectrumMetrics(P, freqs)
			out += metrics.timeMetrics(P,bins)
			out += metrics.oopsMetrics(P,bins)
			out += metrics.highFreqMetrics(P,bins)
			out += metrics.highFreqTemplate(P, bar_, bins)
			out += metrics.highFreqTemplate(P, bar1_, bins)

			hL.append(out)			
		hL = np.array(hL)
		file = open(testOutFile,'w')
		file.write(outHdr+"\n")
		np.savetxt(file,hL,delimiter=',')
		file.close()
		

if __name__ == "__main__":
	main()
