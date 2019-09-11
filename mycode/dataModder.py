import csv
import os
from math import floor
import sys
import numpy as np

expNumber = sys.argv[1]
experiment = 'Experiment_' + expNumber + '/'

#converts 4 to 3
def quaternion2euler3(qx,qy,qz,qw):
    """from tsinghua paper"""
    qx = np.float(qx)
    qy = np.float(qy)
    qz = np.float(qz)
    qw = np.float(qw)
    """convert quaternion tuple to euler angles"""
    x = 2*qx*qz+2*qy*qw
    y = 2*qy*qz-2*qx*qw
    z = 1-2*(qx**2)-2*(qy**2)
    return x,y,z

#cg
# parent_dir = '/home/apurvji/Desktop/apurv/NYU/dataset/'
parent_dir = './360video/tsinghua/'

path = os.path.join(parent_dir+'Formated_Data/',experiment)
folders = [x[0] for x in os.walk(path)]
folders = folders[1:]

videoMetaFile = os.path.join(parent_dir+'Formated_Data/',experiment,'videoMeta.csv')
videoMeta = open(videoMetaFile,'r').readlines()


for folder in folders:
	files = [x[2] for x in os.walk(folder)]
	
	for filename in files[0]:
		print(filename)
		video_ind = int(filename.split("_")[1].split(".")[0])
		video_info = videoMeta[video_ind + 1].split(',')
		fps = int(video_info[3])
		print(fps)
		if filename.startswith("video_") & filename.endswith(".csv"):
			print(filename)
	        	with open(folder + "/" + filename,'r') as csvinput:
			    with open(folder +  "/" + "new_" + filename, 'w') as csvoutput:
			        writer = csv.writer(csvoutput, lineterminator='\n')
			        reader = csv.reader(csvinput)

			        all_ = []
			        row = next(reader)
			        row.append("Frame Index")
			        row.append("xCoordinate")
			        row.append("yCoordinate")
			        row.append("zCoordinate")
			        all_.append(row)

			        for row in reader:
			        	row.append(int(floor(float(row[1])*fps) + 1))
			        	xcord, ycord, zcord = quaternion2euler3(row[2], row[3], row[4], row[5])
			        	row.append(xcord)
			        	row.append(ycord)
			        	row.append(zcord)
			        	all_.append(row)

			        writer.writerows(all_)

