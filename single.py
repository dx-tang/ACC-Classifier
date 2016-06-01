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

def ParsePartTrain(f):
	# X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmp = []
		tmp.extend(columns[PARTAVG:PARTSKEW])
		tmp.extend(columns[RECAVG:CONFRATE])
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
		tmp.extend(columns[RECAVG:CONFRATE])
		X.append(tmp)
		if (columns[FEATURELEN] == 1):
			Y.extend([1])
		elif (columns[FEATURELEN] == 2):
			Y.extend([2])

	return np.array(X), np.array(Y)

def ParsePureTrain(f):
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmp = []
		tmp.extend(columns[RECAVG:HOMECONF])
		tmp.extend(columns[CONFRATE:FEATURELEN])
		X.append(tmp)
		if (columns[FEATURELEN] == 3):
			Y.extend([3])
			if len(columns[FEATURELEN:]) == 2:
				if columns[FEATURELEN+1] == 4:
					X.append(tmp)
					Y.extend([4])
		elif (columns[FEATURELEN] == 4):
			Y.extend([4])

	return np.array(X), np.array(Y)

def ParseIndexTrain(f):
	# X is feature, while Y is label
	X = []
	Y = []
	for line in open(f):
		columns = [float(x) for x in line.strip().split('\t')[FEATURESTART:]]
		tmp = []
		tmp.extend(columns[PARTAVG:RECAVG])
		tmp.extend(columns[LATENCY:READRATE])
		tmp.extend(columns[HOMECONF:CONFRATE])
		X.append(tmp)
		if (columns[FEATURELEN] <= 2):
			Y.extend([0])
		else:
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
def PredictIndex(indexclf, X_test, Y_test):
	tmp=[]
	tmp.extend(X_test[PARTAVG:RECAVG])
	tmp.extend(X_test[LATENCY:READRATE])
	tmp.extend(X_test[HOMECONF:CONFRATE])
	testIndex = []
	testIndex.append(tmp)
	result = indexclf.predict(testIndex)
	ok = 0
	if result[0] == 0:
		for j, y in enumerate(Y_test):
			if (y <= 2):
				ok = 2
				break
	else:
		for j, y in enumerate(Y_test):
			if (y > 2):
				ok = 2
				break
	return result[0], ok

def PredictPure(pureclf, X_test, Y_test):
	tmp=[]
	tmp.extend(X_test[RECAVG:HOMECONF])
	tmp.extend(X_test[CONFRATE:FEATURELEN])
	testPure = []
	testPure.append(tmp)
	result = pureclf.predict(testPure)
	ok = 0
	for j, y in enumerate(Y_test):
		if (y == result[0]):
			ok = 1
			break
	return result[0], ok

def PredictPart(partclf, X_test, Y_test):
	tmp=[]
	tmp.extend(X_test[PARTAVG:PARTSKEW])
	tmp.extend(X_test[RECAVG:CONFRATE])
	testPart = []
	testPart.append(tmp)
	result = partclf.predict(testPart)
	ok = 0
	if (result[0] == 0):
		for j, y in enumerate(Y_test):
			if (y == 0):
				ok = 1
				break
	else:
		for j, y in enumerate(Y_test):
			if (y != 0):
				ok = 2
				break
	print result," ",Y_test," ",partclf.predict_proba(testPart)
	return result[0], ok

def PredictOCC(occclf, X_test, Y_test):
	tmp=[]
	tmp.extend(X_test[RECAVG:CONFRATE])
	testOCC = []
	testOCC.append(tmp)
	result = occclf.predict(testOCC)
	ok = 0
	for j, y in enumerate(Y_test):
		if (y == result[0]):
			ok = 1
			break
	return result[0], ok


def main():
	if (len(sys.argv) < 6):
		print("Five Argument Required: Part-training; OCC-training; Pure-training; Index-training; Tests")
		return

	X_part_train, Y_part_train = ParsePartTrain(sys.argv[1])
	X_occ_train, Y_occ_train = ParseOCCTrain(sys.argv[2])
	X_pure_train, Y_pure_train = ParsePureTrain(sys.argv[3])
	X_index_train, Y_index_train = ParseIndexTrain(sys.argv[4])
	X_test, Y_test = ParseTest(sys.argv[5])

	partclf = tree.DecisionTreeClassifier(max_depth=6)
	#partclf = RandomForestClassifier(max_depth=6, n_estimators=10, max_features=1)
	partclf = partclf.fit(X_part_train, Y_part_train)

	occclf = tree.DecisionTreeClassifier(max_depth=6)
	#occclf = RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1)
	occclf = occclf.fit(X_occ_train, Y_occ_train)
	
	pureclf = tree.DecisionTreeClassifier(max_depth=4)
	pureclf = pureclf.fit(X_pure_train, Y_pure_train)

	indexclf = tree.DecisionTreeClassifier(max_depth=4)
	indexclf = indexclf.fit(X_index_train, Y_index_train)

	count = 0.0
	wrong = 0.0
	occCount = 0.0
	occWrong = 0.0
	partCount = 0.0
	partWrong = 0.0
	indexWrong = 0.0
	pureCount = 0.0
	pureWrong = 0.0
	for i, val in enumerate(X_test):
		count = count + 1
		result, ok = PredictIndex(indexclf, val, Y_test[i])
		if ok == 0: # Wrong
			print i, " ", result, " ",Y_test[i]
			indexWrong = indexWrong+1
			continue
		if result == 0: # Using Partitioned Index
			partCount = partCount + 1
			result, ok = PredictPart(partclf, val, Y_test[i])
			if ok == 0: #Wrong
				partWrong = partWrong + 1
			elif ok == 2: # ok == 2
				occCount = occCount + 1
				result, ok = PredictOCC(occclf, val, Y_test[i])
				if ok == 0: #Wrong
					occWrong = occWrong + 1
		else: # Using Shared Index
			pureCount = pureCount + 1
			result, ok = PredictPure(pureclf, val, Y_test[i])
			if ok == 0:
				print i, " ", result, " ",Y_test[i]
				pureWrong = pureWrong + 1
		
	totalWrong = indexWrong + partWrong + occWrong + pureWrong
	if partCount == 0:
		partCount = 1
	if occCount == 0:
		occCount = 1
	if pureCount == 0:
		pureCount = 1
	print "Total ", count, " ", count - totalWrong, " ",(count - totalWrong)/count
	print "Index ", count, " ", count - indexWrong, " ",(count - indexWrong)/count
	print "Part ", partCount, " ", partCount - partWrong, " ",(partCount - partWrong)/partCount
	print "OCC: ", occCount, " ", occCount - occWrong, " ", (occCount - occWrong)/occCount
	print "PURE: ", pureCount, " ", pureCount - pureWrong, " ", (pureCount - pureWrong)/pureCount

if __name__ == "__main__":
    main()
