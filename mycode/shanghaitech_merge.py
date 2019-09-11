import numpy as np
import cPickle as p
import os
path = '/scratch/wz1219/FoV/data/shanghaitech_one_hot_second'
files= os.listdir(path)
s={}
a = 0
for file in files:
     if not os.path.isdir(file):
          #print(file)
          print(path+"/"+file)
          f = p.load(open(path+"/"+file,'rb'))
          #iter_f = iter(f);

          #for line in iter_f:
          #    str = str + line
          s[file[-5:-2]] = f
          #print(file[-5:-2])
          #a =a +1
          #if file[-5:-2] == '161':
          #    print('161161')
print(s.keys())
#print(a)

p.dump(s,open('/scratch/wz1219/FoV/data/shanghaitech_one_hot_second/shanghaitech_one_hot_second.p','wb'))
