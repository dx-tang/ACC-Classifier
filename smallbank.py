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

PARTCLF = 0
OCCCLF = 1

TP = [0.0, 0.0]
FP = [0.0, 0.0]
TN = [0.0, 0.0]
FN = [0.0, 0.0]

Precision = [0.0, 0.0]
Accuracy = [0.0, 0.0]
Recall = [0.0, 0.0]
F1 = [0.0, 0.0]

def ParsePartTrain(f):
	# X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmp = []
		tmp.extend(columns[PARTAVG:RECAVG])
		tmp.extend(columns[RECAVG:LATENCY])
		tmp.extend(columns[READRATE:CONFRATE])
		X.append(tmp)
		ok = 1
		label = columns[FEATURELEN:]
		if label[0] == 0:
			Y.extend([0])
		else:
			Y.extend([1])
	return np.array(X), np.array(Y)


def ParseOCCTrain(f):
	# X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmp = []
		tmp.extend(columns[RECAVG:LATENCY])
		tmp.extend(columns[READRATE:HOMECONF])
		tmp.extend(columns[CONFRATE:FEATURELEN])
		X.append(tmp)
		if (columns[FEATURELEN] == 1):
			Y.extend([1])
		elif (columns[FEATURELEN] == 2):
			Y.extend([2])

		label = columns[FEATURELEN:]
		if len(label) >= 2:
			if label[1] == 1:
				X.append(tmp)
				Y.extend([1])

	return np.array(X), np.array(Y)

def ParseTest(f):
    # X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmpX = columns[:FEATURELEN]
		tmpY = columns[FEATURELEN:]
		X.append(tmpX)
		Y.append(tmpY)
	return np.array(X), np.array(Y)

# ok = 0 wrong prediction; ok = 1 right prediction; ok = 2 continue prediction
def PredictPart(partclf, X_test, Y_test):
	tmp=[]
	tmp.extend(X_test[PARTAVG:RECAVG])
	tmp.extend(X_test[RECAVG:LATENCY])
	tmp.extend(X_test[READRATE:CONFRATE])
	testPart = []
	testPart.append(tmp)
	result = partclf.predict(testPart)
	ok = 0
	if (result[0] == 0):
		for j, y in enumerate(Y_test):
			if (y == 0):
				TP[PARTCLF] += 1
				ok = 1
				break
		if ok == 0:
			FP[PARTCLF] += 1
	else:
		for j, y in enumerate(Y_test):
			if (y != 0):
				TN[PARTCLF] += 1
				ok = 2
				break
		if ok == 0:
			FN[PARTCLF] += 1
	#print result," ",Y_test," ",partclf.predict_proba(testPart)
	return result[0], ok

def PredictOCC(occclf, X_test, Y_test):
	tmp=[]
	tmp.extend(X_test[RECAVG:LATENCY])
	tmp.extend(X_test[READRATE:HOMECONF])
	tmp.extend(X_test[CONFRATE:FEATURELEN])
	testOCC = []
	testOCC.append(tmp)
	result = occclf.predict(testOCC)
	ok = 0
	for j, y in enumerate(Y_test):
		if (y == result[0]):
			ok = 1
			if result[0] == 2:
				TP[OCCCLF] += 1
			else:
				TN[OCCCLF] += 1
			break
	if ok == 0:
		if result[0] == 2:
			FP[OCCCLF] += 1
		else:
			FN[OCCCLF] += 1
	return result[0], ok


def main():
	if (len(sys.argv) != 4):
		print("Three Arguments Required: Part-training; OCC-training; Tests")
		return

	X_part_train, Y_part_train = ParsePartTrain(sys.argv[1])
	X_occ_train, Y_occ_train = ParseOCCTrain(sys.argv[2])
	X_test, Y_test = ParseTest(sys.argv[3])

	partclf = tree.DecisionTreeClassifier(max_depth=3)
	#partclf = RandomForestClassifier(max_depth=6, n_estimators=10, max_features=1)
	partclf = partclf.fit(X_part_train, Y_part_train)

	occclf = tree.DecisionTreeClassifier(max_depth=4)
	#occclf = RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1)
	occclf = occclf.fit(X_occ_train, Y_occ_train)

	count = 0.0
	wrong = 0.0
	partCount = 0.0
	partWrong = 0.0
	occCount = 0.0
	occWrong = 0.0
	for i, val in enumerate(X_test):
		count = count + 1
		partCount = partCount + 1
		result, ok = PredictPart(partclf, val, Y_test[i])
		if ok == 0: #Wrong
			print i, " ", result, " ",Y_test[i]
			partWrong = partWrong + 1
		elif ok == 2: # ok == 2
			occCount = occCount + 1
			result, ok = PredictOCC(occclf, val, Y_test[i])
			if ok == 0: #Wrong
				#print i, " ", result, " ",Y_test[i]
				occWrong = occWrong + 1
		
	totalWrong = partWrong + occWrong
	if partCount == 0:
		partCount = 1
	if occCount == 0:
		occCount = 1

	for i, _ in enumerate(TP):
		Precision[i] = TP[i]/(TP[i]+FP[i])
		Recall[i] = TP[i]/(TP[i]+FN[i])
		Accuracy[i] = (TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] + FN[i])
		F1[i] = 2*Precision[i]*Recall[i]/(Precision[i] + Recall[i])

	print "Total ", count, " ", count - totalWrong, " ",(count - totalWrong)/count
	print "Part ", partCount, " ", partCount - partWrong, " ",(partCount - partWrong)/partCount
	print "OCC: ", occCount, " ", occCount - occWrong, " ", (occCount - occWrong)/occCount

	print "C1\t", Accuracy[PARTCLF], "\t", Precision[PARTCLF], "\t", Recall[PARTCLF], "\t", F1[PARTCLF]
	print "C2\t", Accuracy[OCCCLF], "\t", Precision[OCCCLF], "\t", Recall[OCCCLF], "\t", F1[OCCCLF]

	print "TP ", TP
	print "TN ", TN
	print "FP ", FP
	print "FN ", FN

if __name__ == "__main__":
    main()
