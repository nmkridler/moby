from classifier import Classify
from sklearn.ensemble import GradientBoostingClassifier

def main():
	baseDir = '/Users/nkridler/Desktop/whale/'

	params = {'max_depth':8, 'subsample':0.5, 'verbose':2, 'random_state':0
		'min_samples_split':20, 'min_samples_leaf':20, 'max_features':30,
		'n_estimators': 500, 'learning_rate': 0.05}
		#'n_estimators': 12000, 'learning_rate': 0.002}
	clf = GradientBoostingClassifier(**params)	
	test = Classify(baseDir+'workspace/trainMetrics.csv')
	test.validate(clf,nFolds=2,featureImportance=True)

if __name__=="__main__":
	main()