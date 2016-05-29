import sys
import pydot
from sklearn.externals.six import StringIO
from sklearn import cross_validation
from IPython.display import Image

from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

FEATURESTART = 5
FEATURELEN = 6
PARTAVG = 0
PARTSKEW = 1
RECAVG = 2
LATENCY = 3
READRATE = 4
CONFRATE = 5


def ParseOCCTrain(f):
    # X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmpX = columns[RECAVG:FEATURELEN]
		tmpY = columns[FEATURELEN:]
		if (tmpY[0] != 0):
			if (len(columns) <= FEATURELEN + 1 or (len(columns) > FEATURELEN + 1 and columns[FEATURELEN+1] != 0)):
				Z = [0, 0]
				X.append(tmpX)
				for y in tmpY:
					Z[int(y) - 1] = 1
				Y.append(Z)

	return np.array(X), np.array(Y)

def ParseTest(f):
    # X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmpX = columns[0:FEATURELEN]
		#tmpX.extend(columns[3:7])
		X.append(tmpX)
		tmpY = columns[FEATURELEN:]
		Y.append(tmpY)
		#addY = []
		#if (len(tmpY) < 2):
		#	if (tmpY[0] == 0):
		#		Y.append([0])
		#	else:
		#		Y.append([1])
		#else:
		#	if (tmpY[0] == 0):
		#		addY.extend([0])
		#	else:
		#		addY.extend([1])
		#	if (tmpY[1] == 0):
		#		addY.extend([0])
		#	else:
		#		addY.extend([1])
		#	Y.append(addY)

	return np.array(X), np.array(Y)

def main():
	if (len(sys.argv) < 3):
		print("One Argument Required; Training Set")
		return

	X_occ_train, Y_occ_train = ParseOCCTrain(sys.argv[1])
	X_test, Y_test = ParseTest(sys.argv[2])

	occclf = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=6))
	occclf = occclf.fit(X_occ_train, Y_occ_train)

	count = 0.0
	wrong = 0.0
	for i, val in enumerate(X_test):
		count = count + 1
		testOCC = []
		ok = 0
		testOCC.append(val[RECAVG:FEATURELEN])
		result = occclf.predict(testOCC)
		#score = occclf.score([val[1:]], [Y_test[i]])
		#for j, y in enumerate(Y_test[i]):
		#	if (y != 0):
		#		if (result[0][y - 1] != 0):
		#			ok = 1
		#			break
		if (result[0][0] == 0):
			for j, y in enumerate(Y_test[i]):
				if (y == 2):
					ok = 1
					break
		elif (result[0][1] == 0):
			for j, y in enumerate(Y_test[i]):
				if (y == 1):
					ok = 1
					break
		else:
			if (len(Y_test) > 1 and Y_test[0] !=0 and Y_test[1] != 0):
				ok = 1
		if (ok == 0):
			print i," ",result[0]," ",Y_test[i]
			wrong = wrong + 1
	print "Total Accuracy ",(count - wrong)/count
	#print (occCount - occWrong)/occCount

if __name__ == "__main__":
    main()
