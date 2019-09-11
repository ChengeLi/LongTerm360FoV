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

def anomaly(user_no, user_ind):
    # for user_ind in range(1,user_num + 1):
        # print("\nUser " + str(user_ind) + " anomalies:")
    filepath = directory + str(user_ind) + '/new_user_' + str(user_no) + '.csv'
    data = pd.read_csv(filepath)
    frame_vals = data['Frame Index'].values
    # print(frame_vals.shape)
    anomalies = frame_vals[:-1] <= frame_vals[1:]
    indices = np.where(anomalies == False)
	#    
    return indices

path = os.path.join('/scratch/ag6925/intern_apurv/NYU/FoV-master/data/Formated_Data/',experiment)
users = [x[0] for x in os.walk(path)]
users = users[1:]

videoMetaFile = os.path.join('/scratch/ag6925/intern_apurv/NYU/FoV-master/data/Formated_Data/',experiment,'videoMeta.csv')
videoMeta = open(videoMetaFile,'r').readlines()


for user in users:
	videos = [x[2] for x in os.walk(user)]

	for video in videos[0]:
		print(video)
		video_ind = int(video.split("_")[1].split(".")[0])
		video_info = videoMeta[video_ind + 1].split(',')
		fps = float(video_info[4])/float(60 * int(video_info[2].split(':')[0]) + int(video_info[2].split(':')[1]))
		# fps = int(user_info[3])
		print(fps)
		if video.startswith("video_") & video.endswith(".csv"):
			print(video)
	        with open(user + "/" + video,'r') as csvinput:
			    with open(user +  "/" + "NEW_" + video, 'w') as csvoutput:
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

