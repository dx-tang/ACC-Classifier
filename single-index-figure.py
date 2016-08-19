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

FEATURESTART = 7
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
		tmp.extend(columns[RECAVG:LATENCY])
		tmp.extend(columns[READRATE:HOMECONF])
		tmp.extend(columns[CONFRATE:FEATURELEN])
		if (columns[FEATURELEN] == 1):
			X.append(tmp)
			Y.extend([1])
		elif (columns[FEATURELEN] == 2):
			X.append(tmp)
			Y.extend([2])

	return np.array(X), np.array(Y)

def main():
	if (len(sys.argv) < 2):
		print("One Argument Required; Training Set; Testing Set")
		return
	X_train, Y_train = ParseTraining(sys.argv[1])
	#X_test, Y_test = ParseTraining(sys.argv[2])
    #X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=99)
    #X_train, X_test, Y_train, Y_test = X, X, Y, Y
    #clf = tree.DecisionTreeClassifier()
	clf = tree.DecisionTreeClassifier(max_depth=3)
    #clf = OneVsRestClassifier(SVC(kernel="linear", C=0.025))
    #clf = RandomForestClassifier(max_depth=6, n_estimators=10, max_features=1)
    #clf = SVC(kernel="linear", C=0.025)
    #clf = AdaBoostClassifier()
    #clf = SVC(gamma=2, C=1)
	clf = clf.fit(X_train, Y_train)


    #feature_names = ["partAvg", "recavg", "latency", "ReadRate"]
	feature_names = ["RECAVG", "READRATE", "CONFRATE"]
    #feature_names = ["partAvg", "recAvg", "recVar", "ReadRate"]
    #feature_names = ["partAvg", "recAvg", "recVar"]
    #feature_names = ["recAvg", "recVar", "Read"]
    #feature_names = ["partAvg", "recVar"]
    ##class_names = ["Partition", "OCC", "2PL"]
	class_names = ["OCC", "2PL"]
	#class_names = ["Partition", "No Partition"]
	dot_data = StringIO()
	tree.export_graphviz(clf, out_file=dot_data,
						feature_names=feature_names,
						class_names=class_names,
						filled=True, rounded=True,
						special_characters=True)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_png("occ.png")
	#print(clf.score(X_test, Y_test))
    #predictArray = clf.predict(X_test)
    #print(predictArray)
    #for i, val in enumerate(predictArray):
    #    if (val != Y_test[i]):
    #	    print i,": ",val," ",Y_test[i]
    

if __name__ == "__main__":
	main()
