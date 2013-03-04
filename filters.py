import random
import numpy as np
import aifc
import scipy.fftpack as spFFT
import cv2

def PSD(s):
	n = s.size
	w = 0.5*(1. - np.cos( 2.*np.pi*np.arange(n)/(n-1)))
	p = spFFT.fftshift(abs(spFFT.fft(s*w)))[:n/2]
	f = spFFT.fftfreq(n)[:n/2]
	return f, p*p	

def CreateMask(img, lower):
	mask = np.zeros(img.shape, dtype='Float32')
	mask[img >= lower] = 1.0
	return mask
	
def Dilate(img, ksize=3):
	kernel = np.ones((ksize,ksize), 'Float32')
	img = cv2.erode(img, kernel)
	return cv2.dilate(img, kernel)