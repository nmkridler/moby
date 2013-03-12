import random
import numpy as np

import cv2
import cv2.cv as cv
from scipy import polyval, polyfit
import pylab as pl

def lineFit(P, freqs, bins):
	m, n = P.shape
	wt = np.array([ P[:,j].argmax() for j in range(n)])
	v = np.array([ P[:,j].max() for j in range(n)])
	y = [ w for w in wt if w > 0]
	x = [ i for i in range(n) if wt[i] > 0]
	weight = [ v[i] for i in range(n) if wt[i] > 0 ]
	z = np.polyfit(x,y,3,w=weight)
	return [zi for zi in z]

def matchTemplate(P, template):
	mf = cv2.matchTemplate(P.astype('Float32'), template, cv2.TM_CCOEFF_NORMED)
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mf)
	return maxVal, maxLoc[0], maxLoc[1]

def ShiftIntegrate(P, x, shifts):
	for i in range(len(x)):
		P[:,x[i]] = np.roll(P[:,x[i]],-shifts[i])
	return np.sum(P,1)

def getMaxPath(P, ksize=3, maxM=15):
	m, n = P.shape
	power = P[:,0].copy()
	index = np.zeros((m,n))
	index[:,0] = np.arange(m)
	for i in range(1,n):
		for j in range(maxM):
			k = index[j,i-1]
			l, r = max(0,k-ksize), min(k+ksize+1,m)
			index[j,i] = P[l:r,i].argmax() + l
			power[j] += P[l:r,i].max()
	v = power[:maxM].argmax()
	return index[v,:]

def slidingWindow(P,inner=3,maxM = 50):
	""""""
	m, n = P.shape
	mval, sval = np.mean(P), np.std(P)
	P[P > mval + sval] = mval + sval
	P[P < mval - sval] = mval - sval
	wInner = np.ones(inner)

	Q = P[:maxM,:].copy()
	for i in range(n):
		Q[:,i] = Q[:,i] - (np.sum(Q[:,i]) - np.convolve(Q[:,i],wInner,'same'))/(maxM - inner)

	return Q

def pathMetrics(P, freqs, bins):
	""""""
	mp = getMaxPath(P,ksize=5)
	u = [P[int(mp[ii]),ii] for ii in range(mp.size)]
	mval, sval = np.mean(u), np.std(u)
	k = [ii for ii in range(mp.size) if u[ii] > mval - 0.25*sval]
	minI, maxI = min(k), max(k)
	minT, maxT, minF, maxF = bins[minI], bins[maxI], freqs[mp[minI]], freqs[mp[maxI]]
	return u + [minT, maxT, maxT - minT, minF, maxF, maxF - minF]

def elongation(m):
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
    return (x + y**0.5) / (x - y**0.5)

def spectrumMetrics(P, freqs):
	""""""
	m, n = P.shape
	cf_ = [np.sum(P[:,i]*freqs)/np.sum(P[:,i]) for i in range(40,n)]

	return cf_

def templateMetrics(P, tmpl, freqs, bins):
	""""""
	maxs, xs, ys = [], [], []
	for k in range(tmpl.size):
		mf, y, x = matchTemplate(P,tmpl.templates[k])
		maxs.append(mf)
		xs.append(x)
		ys.append(y)

	maxMF = np.array(maxs).argmax()
	tmpHit = [0]*(tmpl.size+1)
	tmpHit[maxMF] = 1

	t0, t1 = ys[maxMF], tmpl.n[maxMF] + ys[maxMF]
	f0, f1 = xs[maxMF], tmpl.m[maxMF] + xs[maxMF]
 	maxTime, minTime = bins[t1-1], bins[t0]
	timeLength = maxTime - minTime
	maxFreq, minFreq = freqs[f1-1], freqs[f0] 
	freqLength = maxFreq - minFreq	
	aux = [minFreq, maxFreq, minTime, maxTime, timeLength, freqLength, 
		np.mean(maxs), np.std(maxs), min(maxs), maxs[maxMF]]
	mom_ = cv2.moments(P[f0:f1,t0:t1])
	x_, y_ = mom_['m10']/mom_['m00'], mom_['m01']/mom_['m00']
	m_, s_ = np.mean(P[f0:f1,t0:t1]), np.std(P[f0:f1,t0:t1])
	return maxs + xs + ys + tmpHit + aux + [x_, y_, m_, s_, elongation(mom_)]



