import random
import numpy as np
import cv2
import cv2.cv as cv
import pylab as pl
from scipy.signal import convolve2d

def bounds(x,y):
	return [np.abs(y - (200.+(200.*x[j]/1.5))).argmin() for j in range(x.size)]

def matchTemplate(P, template):
	"""OpenCV matchTemplate"""
	m, n = template.shape
	mf = cv2.matchTemplate(P.astype('Float32'), template, cv2.TM_CCOEFF_NORMED)
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mf)
	return maxVal, maxLoc[0], maxLoc[1]

def slidingWindow(P,inX=3,outX=32,inY=3,outY=64,maxM=50,norm=True):
	"""Demean in the frequency dimension"""
	Q = P.copy()
	m, n = Q.shape
	if norm:
		mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
	wInner = np.ones((inY,inX))
	wOuter = np.ones((outY,outX))
	Q = Q - (convolve2d(Q,wOuter,'same') - convolve2d(Q,wInner,'same'))/(wOuter.size - wInner.size)
	return Q[:maxM,:]

def slidingWindowV(P,inner=3,outer=64,maxM = 50,norm=True):
	"""Demean in the frequency dimension"""
	Q = P.copy()
	m, n = Q.shape
	if norm:
		mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
	wInner = np.ones(inner)
	wOuter = np.ones(outer)
	for i in range(n):
		Q[:,i] = Q[:,i] - (np.convolve(Q[:,i],wOuter,'same') - np.convolve(Q[:,i],wInner,'same'))/(outer - inner)
	return Q[:maxM,:]

def slidingWindowH(P,inner=3,outer=32,maxM=50,norm=True):
	"""Demean in the time dimension"""
	Q = P.copy()
	m, n = Q.shape
	if norm:
		mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
	wInner = np.ones(inner)
	wOuter = np.ones(outer)
	for i in range(maxM):
		Q[i,:] = Q[i,:] - (np.convolve(Q[i,:],wOuter,'same') - np.convolve(Q[i,:],wInner,'same'))/(outer - inner)
	return Q[:maxM,:]

def spectrumMetrics(P, Q, freqs, up_):
	"""Centroids for bin 40+"""
	m, n = P.shape
	cf_ = [np.sum(P[:,i]*freqs)/np.sum(P[:,i]) for i in range(40,n)]
	#cf2 = np.array([Q[:up_[i],i].argmax() for i in range(n)])
	#cf_ = np.convolve(cf_,np.ones(5),'same')/5.
	#cf2_ = [int(np.sum(P[:n,i]*range(n))/np.sum(P[:,i])) for i in range(n)]
	#ratio_ = [ 10.*np.log10(np.sum(P[int(cf2[i]),:])/np.sum(P[:n,i])) for i in range(n)]
	#ratio_ = [np.sum(P[:,i]*(freqs - cf2[i])*(freqs - cf2[i]))/np.sum(P[:,i]) for i in range(n)]
	return cf_ #[cf for cf in cf_] #+ ratio_ 

def timeMetrics(P, b,maxM=50):
	"""Centroids and bandwidth for each time slice"""
	m, n = P.shape
	cf_ = [np.sum(P[i,:]*b)/np.sum(P[i,:]) for i in range(maxM)]
	#cf_ = [cf for cf in np.convolve(cf_,np.ones(5),'same')/5.]
	bw_ = [np.sum(P[i,:]*(b - cf_[i])*(b - cf_[i]))/np.sum(P[i,:]) for i in range(maxM)]
	return cf_ + bw_	

def oopsMetrics(P, b,maxM=50):
	"""Centroids and bandwidth for each time slice"""
	cf_ = [np.sum(P[i,:]*b)/np.sum(P[:,i]) for i in range(maxM)]
	#cf_ = [cf for cf in np.convolve(cf_,np.ones(5),'same')/5.]
	#bw_ = [np.sum(P[i,:]*(b - cf_[i])*(b - cf_[i]))/np.sum(P[i,:]) for i in range(maxM)]
	return cf_ #+ bw_
		
def highFreqTemplate(P, tmpl, bins):
	"""Look for bands at high frequencies"""
	Q = slidingWindowH(P,inner=7,maxM=50,norm=True)[38:,:]	
	mf = cv2.matchTemplate(Q.astype('Float32'), tmpl, cv2.TM_CCOEFF_NORMED)
	return [mf.max()]

def highFreqMetrics(P, bins):
	"""Look for variation at high frequencies"""
	Q = slidingWindowH(P,inner=7,maxM=50,norm=True)[25:,:]	
	m, n = Q.shape
	cf_ = np.empty(m)
	bw_ = np.empty(m)
	for i in range(m):
		mQ = Q[i,:]
		min_, max_ = mQ.min(), mQ.max()
		mQ = (mQ - min_)/(max_ - min_)
		cf_[i] = np.sum(mQ*bins)/np.sum(mQ)
		bw_[i] = np.sqrt(np.sum(mQ*(bins-cf_[i])*(bins-cf_[i]))/np.sum(mQ))

	mQ = np.sum(Q[14:,:],0)
	min_, max_ = mQ.min(), mQ.max()
	mQ = (mQ - min_)/(max_ - min_)
	cfM_ = np.sum(mQ*bins)/np.sum(mQ)
	bwM_ = np.sqrt(np.sum(mQ*(bins - cfM_)*(bins -cfM_))/np.sum(mQ))

	return [np.std(cf_), np.mean(bw_), cfM_, bwM_]


def templateMetrics(P, tmpl):
	"""Match Template metric calculations"""
	maxs, xs, ys = [], [], []
	for k in range(tmpl.size):
		mf, y, x  = matchTemplate(P,tmpl.templates[k])
		maxs.append(mf)
		xs.append(x)
		ys.append(y)
	return maxs + xs + ys



