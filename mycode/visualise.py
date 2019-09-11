import _pickle as pickle
import sys
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
import time
from matplotlib import pyplot as plt
import importlib
import string
from sklearn import metrics

from fancyimpute import (
    BiScaler,
    KNN,
    NuclearNormMinimization,
    SoftImpute,
    SimpleFill,
    MatrixFactorization,
    IterativeSVD,
    MICE
)

expNumber = sys.argv[1]
file_Name = "/home/apurvji/Desktop/apurv/NYU/dataset/exp_" + expNumber + "_matrix.p"
fileObject = open(file_Name,'r')
data = pickle.load(fileObject)
print(type(data))
fileObject.close()

def id_generator(size=2, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


video = random.choice(range(9))
mat = data[video]
rows, columns = mat.shape
query_usr = random.choice(range(rows))	#+1 is the user number
col = random.choice(range(columns - 600))
target = mat.iloc[:,col:(col + 600)].copy()
original1 = target.apply(lambda x: x.str[0]).copy()
original2 = target.apply(lambda x: x.str[1]).copy()
original3 = target.apply(lambda x: x.str[2]).copy()

original1 = original1.fillna(method = 'ffill')
original2 = original2.fillna(method = 'ffill')
original3 = original3.fillna(method = 'ffill')

orig_normed_matrix = np.sqrt(original1.values**2 + original2.values**2 + original3.values**2)

copy1 = original1.values.copy()
copy2 = original2.values.copy()
copy3 = original3.values.copy()

copy1 = copy1/orig_normed_matrix
copy2 = copy2/orig_normed_matrix
copy3 = copy3/orig_normed_matrix

orig1 = copy1.copy()
orig2 = copy2.copy()
orig3 = copy3.copy()

copy1[query_usr,300:600] = np.nan
copy2[query_usr,300:600] = np.nan
copy3[query_usr,300:600] = np.nan

print("\n -------------------- \n")

start = time.time()

completed1SI = SoftImpute().complete(copy1)
completed2SI = SoftImpute().complete(copy2)
completed3SI = SoftImpute().complete(copy3)

normed_matrix = np.sqrt(completed1SI**2 + completed2SI**2 + completed3SI**2)
completed1SI = completed1SI/normed_matrix
completed2SI = completed2SI/normed_matrix
completed3SI = completed3SI/normed_matrix

orig1 = orig1 + 1
orig2 = orig2 + 1
orig3 = orig3 + 1

copy1 = copy1 + 1
copy2 = copy2 + 1
copy3 = copy3 + 1

completed1SI = completed1SI + 1
completed2SI = completed2SI + 1
completed3SI = completed3SI + 1

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
import sys
import pandas as pd
# dataname = sys.argv[1]

eyedee = id_generator()
letter = ['x', 'y', 'z']
#matrix with nans
copies = [copy1, copy2, copy3]
for idx in range(len(copies)):
	mat = copies[idx]
	b = np.ceil(mat * 255)
	b = b.astype('uint8')
	c = cv2.resize(b, (600, 480), interpolation = cv2.INTER_NEAREST)
	d = cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)
	d[query_usr*10 : query_usr*10 + 10, 300:600, 0] = 0
	d[query_usr*10 : query_usr*10 + 10, 300:600, 1] = 0
	d[query_usr*10 : query_usr*10 + 10, 300:600, 2] = 255 
	# cv2.imshow('a',d)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('/home/apurvji/Desktop/visualisations/' + eyedee + 'copy_' + letter[idx] + '.png', d)

#matrix with originals
originals = [original1.values, original2.values, original3.values]
for idx in range(len(originals)):
	mat = originals[idx]
	b = np.ceil(mat * 255)
	b = b.astype('uint8')
	c = cv2.resize(b, (600, 480), interpolation = cv2.INTER_NEAREST)
	d = cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)
	# d[query_usr*10 : query_usr*10 + 10, 300:600, 0] = 0
	# d[query_usr*10 : query_usr*10 + 10, 300:600, 1] = 255
	# d[query_usr*10 : query_usr*10 + 10, 300:600, 2] = 0 
	# cv2.imshow('a',d)
	d[query_usr*10 - 3, 300:600, 0] = 0
	d[query_usr*10 - 3, 300:600, 1] = 255
	d[query_usr*10 - 3, 300:600, 2] = 0 
	
	d[query_usr*10 + 10 + 3, 300:600, 0] = 0
	d[query_usr*10 + 10 + 3, 300:600, 1] = 255
	d[query_usr*10 + 10 + 3, 300:600, 2] = 0 
	
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 300, 0] = 0
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 300, 1] = 255
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 300, 2] = 0
	

	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 599, 0] = 0
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 599, 1] = 255
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 599, 2] = 0

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('/home/apurvji/Desktop/visualisations/' + eyedee + 'orig_' + letter[idx] + '.png', d)

#matrix with completed values
completed = [completed1SI, completed2SI, completed3SI]
for idx in range(len(completed)):
	mat = completed[idx]
	b = np.ceil(mat * 255)
	b = b.astype('uint8')
	c = cv2.resize(b, (600, 480), interpolation = cv2.INTER_NEAREST)
	d = cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)

	d[query_usr*10 - 3, 300:600, 0] = 255
	d[query_usr*10 - 3, 300:600, 1] = 0
	d[query_usr*10 - 3, 300:600, 2] = 0
	
	d[query_usr*10 + 10 + 3, 300:600, 0] = 255
	d[query_usr*10 + 10 + 3, 300:600, 1] = 0
	d[query_usr*10 + 10 + 3, 300:600, 2] = 0 

	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 300, 0] = 255
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 300, 1] = 0
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 300, 2] = 0	

	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 599, 0] = 255
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 599, 1] = 0
	d[query_usr*10 - 3: query_usr*10 + 10 + 3, 599, 2] = 0

	# cv2.imshow('a',d)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('/home/apurvji/Desktop/visualisations/' + eyedee + 'compl_' + letter[idx] + '.png', d)
