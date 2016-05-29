import sys
import numpy as np

START = 5
FEATURELEN = 7

def main():
	if (len(sys.argv) < 5):
		print("One Argument Required; Training Set")
		return
	wf = open(sys.argv[4], 'w')
	r1 = open(sys.argv[1])
	r2 = open(sys.argv[2])
	r3 = open(sys.argv[3])
	for line1 in r1:
		line2 = r2.readline()
		line3 = r3.readline()
		column1 = [int(x) for x in line1.strip().split('\t')[START+FEATURELEN:]]
		column2 = [int(x) for x in line2.strip().split('\t')[START+FEATURELEN:]]
		column3 = [int(x) for x in line3.strip().split('\t')[START+FEATURELEN:]]
		count = 0
		if (column1[0] == 0):
			count = count + 1
		if (column2[0] == 0):
			count = count + 1
		if (column3[0] == 0):
			count = count + 1
		if (count > 1):
			if (column1[0] == 0):
				wf.write(line1)
			elif (column2[0] == 0):
				wf.write(line2)
			elif (column3[0] == 0):
				wf.write(line3)
			continue
		# OCC or 2PL
		count = 0
		if (column1[0] == 1):
			count = count + 1
		if (column2[0] == 1):
			count = count + 1
		if (column3[0] == 1):
			count = count + 1
		if (count > 1):
			if (column1[0] == 1):
				wf.write(line1)
			elif (column2[0] == 1):
				wf.write(line2)
			elif (column3[0] == 1):
				wf.write(line3)
		else:
			if (column1[0] == 2):
				wf.write(line1)
			elif (column2[0] == 2):
				wf.write(line2)
			elif (column3[0] == 2):
				wf.write(line3)
	wf.close()
	r1.close()
	r2.close()
	r3.close()

if __name__ == "__main__":
    main()
