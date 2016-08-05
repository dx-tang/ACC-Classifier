import sys

import pydot
from sklearn.externals.six import StringIO
from sklearn import cross_validation
from IPython.display import Image
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
import math

FEATURESTART = 5
FEATURELEN = 7
PARTAVG = 0
PARTSKEW = 1
RECAVG = 2
LATENCY = 3
READRATE = 4
HOMECONF = 5
CONFRATE = 6


def ParseTraining(f):
	# X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmp = []
		tmp.extend(columns[PARTAVG:RECAVG])
		tmp.extend(columns[LATENCY:READRATE])
		tmp.extend(columns[HOMECONF:CONFRATE])
		ok1 = 0
		ok2 = 0
		for _, y in enumerate(columns[FEATURELEN:]):
			if y <= 2:
				ok1 = 1
			if y > 2:
				ok2 = 1
		if ok1 == 1:
			X.append(tmp)
			Y.extend([0])
		if columns[FEATURELEN] > 2:
			X.append(tmp)
			Y.extend([1])

	return np.array(X), np.array(Y)


def main():
	if (len(sys.argv) < 2):
		print("One Argument Required; Training Set")
		return
	X_train, Y_train = ParseTraining(sys.argv[1])
	#X_test, Y_test = ParseTesting(sys.argv[2])
    #X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=99)
    #X_train, X_test, Y_train, Y_test = X, X, Y, Y
    #clf = tree.DecisionTreeClassifier()
	clf = tree.DecisionTreeClassifier(max_depth=4)
    #clf = OneVsRestClassifier(SVC(kernel="linear", C=0.025))
	#clf = RandomForestClassifier(max_depth=6, n_estimators=10, max_features=1)
    #clf = SVC(kernel="linear", C=0.025)
    #clf = AdaBoostClassifier()
    #clf = SVC(gamma=2, C=1)
	clf = clf.fit(X_train, Y_train)


    #feature_names = ["partAvg", "recavg", "latency", "ReadRate"]
	feature_names = ["partavg", "partvar", "latency", "homeconf"]
    #feature_names = ["partAvg", "recAvg", "recVar", "ReadRate"]
    #feature_names = ["partAvg", "recAvg", "recVar"]
    #feature_names = ["recAvg", "recVar", "Read"]
    #feature_names = ["partAvg", "recVar"]
    ##class_names = ["Partition", "OCC", "2PL"]
    #class_names = ["OCC", "2PL"]
	class_names = ["Part", "Share"]
	dot_data = StringIO()
	tree.export_graphviz(clf, out_file=dot_data,
						feature_names=feature_names,
						class_names=class_names,
						filled=True, rounded=True,
						special_characters=True)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_png("partition.png")
	print clf.predict_proba([[0,0.4,144,0]])
	#print(clf.score(X_test, Y_test))
    #predictArray = clf.predict(X_test)
    #print(predictArray)
    #for i, val in enumerate(predictArray):
    #    if (val != Y_test[i]):
    #	    print i,": ",val," ",Y_test[i]
    

if __name__ == "__main__":
	main()
