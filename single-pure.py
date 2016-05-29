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

FEATURESTART = 7
FEATURELEN = 7
PARTAVG = 0
PARTSKEW = 1
RECAVG = 2
LATENCY = 3
READRATE = 4
HOMECONF = 5
CONFRATE = 6

def ParseOCCTrain(f):
	# X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		if (columns[FEATURELEN] != 0):
			if (len(columns) <= FEATURELEN + 1 or (len(columns) > FEATURELEN + 1 and columns[FEATURELEN+1] != 0)):
				tmp = []
				tmp.extend(columns[RECAVG:HOMECONF])
				tmp.extend(columns[CONFRATE:FEATURELEN])
				#tmp.extend(columns[5:7])
				X.append(tmp)
				if (columns[FEATURELEN] == 3):
					Y.extend([1])
				elif (columns[FEATURELEN] == 4):
					Y.extend([2])

	return np.array(X), np.array(Y)

def ParseTest(f):
    # X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		#tmpX = columns[0:1]
		#tmpX.extend(columns[3:7])
		tmpX = columns[:FEATURELEN]
		tmpY = []
		for i, y in enumerate(columns[FEATURELEN:]):
			if y == 3:
				tmpY.extend([1])
			elif y == 4:
				tmpY.extend([2])
		if len(tmpY) >= 1:
			X.append(tmpX)
			Y.append(tmpY)
	return np.array(X), np.array(Y)

def main():
	if (len(sys.argv) < 3):
		print("Two Argument Required: Training-pure Set; Test Set")
		return

	X_occ_train, Y_occ_train = ParseOCCTrain(sys.argv[1])
	X_test, Y_test = ParseTest(sys.argv[2])

	occclf = tree.DecisionTreeClassifier(max_depth=6)
	occclf = occclf.fit(X_occ_train, Y_occ_train)

	count = 0.0
	wrong = 0.0
	for i, val in enumerate(X_test):
		count = count + 1
		testOCC = []
		ok = 0
		tmpVal = []
		tmpVal.extend(val[RECAVG:HOMECONF])
		tmpVal.extend(val[CONFRATE:FEATURELEN])
		#testOCC.append(val[RECAVG:FEATURELEN])
		testOCC.append(tmpVal)
		result = occclf.predict(testOCC)
		#score = occclf.score([val[1:]], [Y_test[i]])
		#for j, y in enumerate(Y_test[i]):
		#	if (y != 0):
		#		if (result[0][y - 1] != 0):
		#			ok = 1
		#			break
		if (result[0] == 1):
			for j, y in enumerate(Y_test[i]):
				if (y == 1):
					ok = 1
					break
		elif (result[0] == 2):
			for j, y in enumerate(Y_test[i]):
				if (y == 2):
					ok = 1
					break
		if (ok == 0):
			print i," ",result[0]," ",Y_test[i]
			wrong = wrong + 1
	print "TotalCount: ",count," Wrong: ",wrong
	print "Total: ",(count - wrong)/count
if __name__ == "__main__":
    main()
