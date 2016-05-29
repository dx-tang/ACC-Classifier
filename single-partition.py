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

START = 7
FEATURELEN = 6

def ParseTraining(f):
	# X is feature, while Y is label
    X = []
    Y = []
    for line in open(f):
	columns = [float(x) for x in line.strip().split('\t')[START:]]
	tmp = []
        tmp.extend(columns[0:1])
        #tmp.extend(columns[1:2])
        #tmp.extend([math.ceil(columns[3])])
        tmp.extend(columns[2:6])
	X.append(tmp)
        #if (columns[FEATURELEN] == 0 or (len(columns) > FEATURELEN + 1 and columns[FEATURELEN+1] == 0)):
        if (columns[FEATURELEN] == 0):
	    Y.extend([0])
        else:
	    Y.extend([1])
	#else:
	#    Y.extend([2])
    return np.array(X), np.array(Y)

def main():
    if (len(sys.argv) < 2):
        print("One Argument Required; Training Set")
        return
    X_train, Y_train = ParseTraining(sys.argv[1])
    #X_test, Y_test = ParseTraining(sys.argv[2])
    #X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=99)
    #X_train, X_test, Y_train, Y_test = X, X, Y, Y
    #clf = tree.DecisionTreeClassifier()
    clf = tree.DecisionTreeClassifier(max_depth=6)
    #clf = OneVsRestClassifier(SVC(kernel="linear", C=0.025))
    #clf = RandomForestClassifier(max_depth=6, n_estimators=10, max_features=1)
    #clf = SVC(kernel="linear", C=0.025)
    #clf = AdaBoostClassifier()
    #clf = SVC(gamma=2, C=1)
    clf = clf.fit(X_train, Y_train)


    #feature_names = ["partAvg", "recavg", "latency", "ReadRate"]
    feature_names = ["partConf", "recAvg", "latency", "ReadRate", "confRate"]
    #feature_names = ["partAvg", "recAvg", "recVar", "ReadRate"]
    #feature_names = ["partAvg", "recAvg", "recVar"]
    #feature_names = ["recAvg", "recVar", "Read"]
    #feature_names = ["partAvg", "recVar"]
    ##class_names = ["Partition", "OCC", "2PL"]
    #class_names = ["OCC", "2PL"]
    class_names = ["Partition", "No Partition"]
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_names,
                         class_names=class_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("partition.png")
    #print(clf.score(X_test, Y_test))
    #predictArray = clf.predict(X_test)
    #print(predictArray)
    #for i, val in enumerate(predictArray):
    #    if (val != Y_test[i]):
    #	    print i,": ",val," ",Y_test[i]
    

if __name__ == "__main__":
    main()
