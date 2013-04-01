import classifier
reload(classifier)
from classifier import Classify
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
def main():
	baseDir = '/Users/nkridler/Desktop/whale/'
	if True:
		params = {'max_depth':10, 'subsample':0.5, 'verbose':2,
			'min_samples_split':20, 'min_samples_leaf':20,
			'n_estimators': 500, 'learning_rate': 0.05, 'max_features':30}
			#'n_estimators': 12000, 'learning_rate': 0.002, 'max_features':30}
		clf = GradientBoostingClassifier(**params)	
	else:
		params = {'n_estimators':500, 'n_jobs':-1, 'verbose':True,
				 'compute_importances': True}
		clf = RandomForestClassifier(**params)
	#clf = SVC(probability=True,class_weight="auto")
	test = Classify(baseDir+'workspace/baseTrain7.csv')
	test.validate(clf,nFolds=2,featureImportance=True)
	#test.testAndOutput(clf=clf,testFile=baseDir+'workspace/baseTest7.csv',outfile='401_4.sub')

if __name__=="__main__":
	main()