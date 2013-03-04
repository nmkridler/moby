import random
import numpy as np

import cv2
import cv2.cv as cv
from scipy import polyval, polyfit


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

def slidingWindow(P,inner=7,maxM = 50):
	""""""
	m, n = P.shape
	mval, sval = np.mean(P), np.std(P)
	P[P > mval + sval] = mval + sval
	P[P < mval - sval] = mval - sval

	Q = np.zeros((maxM,n))
	wInner = np.ones(inner)
	
	for i in range(maxM):
		Q[i,:] = P[i,:] - (np.sum(P[i,:]) - np.convolve(P[i,:],wInner,'same'))/(n - inner)

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

	maxTime, minTime = bins[tmpl.n[maxMF] + ys[maxMF] - 1], bins[ys[maxMF]]
	timeLength = maxTime - minTime
	maxFreq, minFreq = freqs[tmpl.m[maxMF] + xs[maxMF] - 1], freqs[xs[maxMF]] 
	freqLength = maxFreq - minFreq	
	aux = [minFreq, maxFreq, minTime, maxTime, timeLength, freqLength]
	return maxs + xs + ys + tmpHit + aux



