import numpy as np
import pylab as pl
from scipy.stats import gaussian_kde
from matplotlib import mlab
from sklearn.metrics import roc_curve, auc

def PlotROC(truth, prediction):
	"""Plot a roc curve"""
	fpr, tpr, thresholds = roc_curve(truth, prediction)

	roc_auc = auc(fpr,tpr)
	print "Area under the curve: %f" % roc_auc
	pl.semilogx(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, lw=3)
	pl.ylim([0.0, 1.0])
	pl.xlim([0.0, 1.0])
	pl.xlabel('PFA')
	pl.ylabel('PD')
	pl.legend(loc="lower right")
	return
	
def PlotDensity(prediction, labelStr, minval=None, maxval=None):
	"""Density plot"""
	if minval != None and maxval != None:
		prediction = (prediction - minval)/(maxval - minval)

	density = gaussian_kde(prediction)
	xs = np.linspace(0,1,200)
	density.covariance_factor = lambda : .1
	density._compute_covariance()
	pl.plot(xs, density(xs), lw=3, label=labelStr)
	pl.legend(loc="upper right")

def PlotSpecgram(P, freqs, bins):
	"""Spectrogram"""
	Z = np.flipud(P)

	xextent = 0, np.amax(bins)
	xmin, xmax = xextent
	extent = xmin, xmax, freqs[0], freqs[-1]

	im = pl.imshow(Z, extent=extent)
	pl.axis('auto')
	pl.xlim([0.0, bins[-1]])
	pl.ylim([0, 400])

	