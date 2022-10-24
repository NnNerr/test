from ctypes import util
import numpy as np
import soundfile as sf
import os,json
import utils
from PIL import Image as img
import random
N = 561*64  #视频数
n = 0
j = 0
k = 0
seq = np.linspace(0,N-1,N)
random.shuffle(seq)
print(seq)
train_set = np.zeros((N,56,56,3),dtype = 'uint8')
id_set = np.zeros((N),dtype= 'uint8')
for i in range(20):
    file_path = f'./train/ID{i+1}'
    for file_name in os.listdir(file_path):
        ## 读取MP4文件中的视频,可以用任意其他的读写库
        print(os.path.join(file_path,file_name))
        video_frames,video_fps = utils.read_video(os.path.join(file_path,file_name))
        #train_set[n] = np.uint8(video_frames[:64])
        k += 1
        ## 做一些处理
        #n += 1
        for j in range(0,96,3):
            rec_rgb = img.fromarray(video_frames[j].astype(np.uint8),'RGB')
            rgb = rec_rgb.resize((75,75))
            train_set[int(seq[n])] = np.uint8(np.array(rgb)[0:56,9:65])
            rec_rgb = img.fromarray(np.array(rgb)[0:56,9:65].astype(np.uint8),'RGB')
            #if file_name == '024.mp4' and j == 12:
            #    rec_rgb.show()
            rgb = np.fliplr(rgb)
            train_set[int(seq[n+1])] = np.uint8(np.array(rgb)[0:56,9:65])
            id_set[int(seq[n])] = i
            id_set[int(seq[n+1])] = i
            n += 2
print(k,n)
print(train_set)
print(id_set)
np.save('train_set',train_set)
np.save('id_set',id_set)