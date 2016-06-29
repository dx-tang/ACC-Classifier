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

FEATURESTART = 5
FEATURELEN = 7
PARTAVG = 0
PARTSKEW = 1
RECAVG = 2
LATENCY = 3
READRATE = 4
HOMECONF = 5
CONFRATE = 6

def ParseTrain(f):
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
		if columns[FEATURELEN] > 2:
			X.append(tmp)
			Y.extend([1])
		else:
			X.append(tmp)
			Y.extend([0])

	return np.array(X), np.array(Y)

def ParseTest(f):
    # X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmpX = columns[:FEATURELEN]
		X.append(tmpX)
		tmpY = columns[FEATURELEN:]
		for i, y in enumerate(tmpY):
			if y <=2:
				tmpY[i] = 0
			else:
				tmpY[i] = 1
		Y.append(tmpY)

	return np.array(X), np.array(Y)

def main():
	if (len(sys.argv) < 3):
		print("Two Argument Required: Training Set; Test Set")
		return

	X_train, Y_train = ParseTrain(sys.argv[1])
	X_test, Y_test = ParseTest(sys.argv[2])

	indexclf = tree.DecisionTreeClassifier(max_depth=6)
	#indexclf = RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1)
	indexclf = indexclf.fit(X_train, Y_train)

	count = 0.0
	wrong = 0.0
	for i, val in enumerate(X_test):
		count = count + 1
		testOCC = []
		ok = 0
		tmp=[]
		tmp.extend(val[PARTAVG:RECAVG])
		tmp.extend(val[LATENCY:READRATE])
		tmp.extend(val[HOMECONF:CONFRATE])
		testOCC.append(tmp)
		result = indexclf.predict(testOCC)
		#score = occclf.score([val[1:]], [Y_test[i]])
		#for j, y in enumerate(Y_test[i]):
		#	if (y != 0):
		#		if (result[0][y - 1] != 0):
		#			ok = 1
		#			break
		if (result[0] == 0):
			for j, y in enumerate(Y_test[i]):
				if (y == 0):
					ok = 1
					break
		elif (result[0] == 1):
			for j, y in enumerate(Y_test[i]):
				if (y == 1):
					ok = 1
					break
		if (ok == 0):
			print i," ",result[0]," ",Y_test[i]
			wrong = wrong + 1
	print "Total: ",(count - wrong)/count

if __name__ == "__main__":
    main()
