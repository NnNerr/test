import numpy as np
import soundfile as sf
import os,json
import utils
import nussl

from PIL import Image as img
import torch
from torch._C import dtype
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import random

import os
import utils
import pickle
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc
from python_speech_features import delta

import librosa
import seperator


def extract_features(audio, rate):

    mfcc_feature = mfcc(
                        audio,
                        rate,
                        winlen = 0.025,
                        winstep = 0.01,
                        numcep = 20,
                        nfilt = 30,
                        nfft = 512,
                        appendEnergy = True)


    mfcc_feature  = preprocessing.scale(mfcc_feature)
    deltas        = delta(mfcc_feature, 2)
    double_deltas = delta(deltas, 2)
    combined      = np.hstack((mfcc_feature, deltas, double_deltas))
    return combined
def takeFirst(elem):
    return elem[0]
db = {}
for fname in [fname for fname in os.listdir('./') if fname.endswith('.gmm')]:
    speaker     = fname.split('.')[0]
    model       = pickle.load( open(os.path.join('./', fname), 'rb'))
    db[speaker] = model


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1),#(16,54,54)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))#(16,27,27)
    
        self.layer2 = nn.Sequential( 
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),#(32,26,26)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))#(32,13,13)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),#(64,12,12)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))#(64,5,5)

        
        self.fc = nn.Sequential(
            nn.Linear(64*5*5,20))
        
    def forward(self,x):
        x = self.layer1(x)  #(b,3,224,224),(3,8,4,4)——(b,8,111,111)  
        x = self.layer2(x)  #(b,8,111,111) ,(8,16,4,4) ——（b,16,55,55）——（b,16,28,28）
        x = self.layer3(x)  #(b,16,28,28）,(16,32,4,4)) ——(b,32,13,13)
        x = x.view(x.size(0),-1) #(b,64,3,3)——(b,64*3*3)
        x = self.fc(x)  #(b,64*3*3)*(64*3*3,256)——(b,256)——(b,256)*(256,64)——(b,64)——(b,64)*(64,20)——(b,20)
        return x 
cnn = torch.load('mycnn_8.pkl')
def test_task1(video_path):
    # 测试1
    result_dict = {}
    for file_name in os.listdir(video_path):
        ## 读取MP4文件中的视频,可以用任意其他的读写库
        
        video_frames,video_fps = utils.read_video(os.path.join(video_path,file_name))
        ## 做一些处理
        ID = []
        for i in range(10,71,3):
            rec_rgb1 = img.fromarray(video_frames[i].astype(np.uint8),'RGB')
            test = np.zeros((1,56,56,3))
            rgb1 = rec_rgb1.resize((75,75))
            test[0] = np.uint8(np.array(rgb1)[0:56,9:65])
            test1 = np.transpose(test,(0,3,1,2))
            test1 = torch.tensor(test1,dtype=torch.float32)
            output = cnn(test1)
            ID.append(int(torch.argmax(output[0,:]))+1)
        ## 返回一个ID
        ser = np.bincount(np.array(ID)) 
        ID_return = int(np.argmax(ser))
        #print(file_name,':',ID_return)
        result_dict[file_name]=utils.ID_dict[ID_return]

    return result_dict

def test_task2(wav_path):
    # 测试2
    result_dict = {}
    for file_name in os.listdir(wav_path):
        ## 读取WAV文件中的音频,可以用任意其他的读写库
        audio_trace11 = utils.read_audio(os.path.join(wav_path,file_name),sr=16000)
        vector1 = extract_features(audio_trace11,16000)

        log_likelihood = []
        for speaker, model in db.items():
            gmm                     = model
            scores                  = gmm.score(vector1)
            log_likelihood.append((round(scores, 3),speaker))
        log_likelihood.sort(key=takeFirst,reverse=True)
            ## 做一些处理

        ## 返回一个ID
        result_dict[file_name]=utils.ID_dict[int(log_likelihood[0][1])]

    return result_dict
def test_task3(video_path,result_path):
    # 测试3
    if os.path.isdir(result_path):
        print('warning: using existed path as result_path')
    else:
        os.mkdir(result_path)
    for file_name in os.listdir(video_path):
        ## 读MP4中的图像和音频数据，例如：
        idx = file_name[-7:-4]  # 提取出序号：001, 002, 003.....

        video_frames,video_fps= utils.read_video(os.path.join(video_path,file_name))
        deal_len = (utils.read_audio(os.path.join(video_path,file_name),sr=44100)).shape[0]
        audio_trace,_ = librosa.load(os.path.join(video_path,file_name),sr=44100)

        ## 做一些处理



        ID1 = []
        ID2 = []
        ID3 = []
        for i in range(10,71,3):
            rec_rgb1 = img.fromarray(video_frames[i,:,0:224].astype(np.uint8),'RGB')
            test = np.zeros((1,56,56,3))
            rgb1 = rec_rgb1.resize((75,75))
            test[0] = np.uint8(np.array(rgb1)[0:56,9:65])
            cnn = torch.load('mycnn_8.pkl')
            test1 = np.transpose(test,(0,3,1,2))
            test1 = torch.tensor(test1,dtype=torch.float32)
            output = cnn(test1)
            ID1.append(int(torch.argmax(output[0,:]))+1)

            rec_rgb1 = img.fromarray(video_frames[i,:,224:448].astype(np.uint8),'RGB')
            test = np.zeros((1,56,56,3))
            rgb1 = rec_rgb1.resize((75,75))
            test[0] = np.uint8(np.array(rgb1)[0:56,9:65])
            cnn = torch.load('mycnn_8.pkl')
            test1 = np.transpose(test,(0,3,1,2))
            test1 = torch.tensor(test1,dtype=torch.float32)
            output = cnn(test1)
            ID2.append(int(torch.argmax(output[0,:]))+1)

            rec_rgb1 = img.fromarray(video_frames[i,:,448:672].astype(np.uint8),'RGB')
            test = np.zeros((1,56,56,3))
            rgb1 = rec_rgb1.resize((75,75))
            test[0] = np.uint8(np.array(rgb1)[0:56,9:65])
            cnn = torch.load('mycnn_8.pkl')
            test1 = np.transpose(test,(0,3,1,2))
            test1 = torch.tensor(test1,dtype=torch.float32)
            output = cnn(test1)
            ID3.append(int(torch.argmax(output[0,:]))+1)
        ## 返回一个ID
        ser = np.bincount(np.array(ID1)) 
        ID_left = int(np.argmax(ser))
        ser = np.bincount(np.array(ID2)) 
        ID_middle = int(np.argmax(ser))
        ser = np.bincount(np.array(ID3)) 
        ID_right = int(np.argmax(ser))           #获得左中右的ID


        audio_left = seperator.seperator(ID_left,audio_trace)
        audio_middle = seperator.seperator(ID_middle,audio_trace)
        audio_right = seperator.seperator(ID_right,audio_trace)

        #audio_left = audio_trace
        #audio_middle = audio_trace
        #audio_right = audio_trace

        len = np.size(audio_left)
        delta = deal_len - len
        a = int(delta/2)
        audio_left = np.pad(audio_left,((a,delta - a)),"constant")
        audio_middle = np.pad(audio_middle,((a,delta - a)),"constant")
        audio_right = np.pad(audio_right,((a,delta - a)),"constant")


        
        ## 输出结果到result_path
        sf.write(os.path.join(result_path,idx+'_left.wav'),   audio_left, 44100)
        sf.write(os.path.join(result_path,idx+'_middle.wav'), audio_middle, 44100)
        sf.write(os.path.join(result_path,idx+'_right.wav'),  audio_right, 44100)

if __name__=='__main__':

    # testing task1
    with open('./test_offline/task1_gt.json','r') as f:
        task1_gt = json.load(f)
    task1_pred = test_task1('./test_offline/task1')
    task1_acc = utils.calc_accuracy(task1_gt,task1_pred)
    print('accuracy for task1 is:',task1_acc)   

    # testing task2
    with open('./test_offline/task2_gt.json','r') as f:
        task2_gt = json.load(f)
    task2_pred = test_task2('./test_offline/task2')
    task2_acc = utils.calc_accuracy(task2_gt,task2_pred)
    print('accuracy for task2 is:',task2_acc)   

    # testing task3
    test_task3('./test_offline/task3','./test_offline/task3_estimate')
    task3_SISDR_blind = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=True)  # 盲分离
    print('strength-averaged SISDR_blind for task3 is:',task3_SISDR_blind)
    task3_SISDR_match = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=False) # 定位分离
    print('strength-averaged SISDR_match for task3 is: ',task3_SISDR_match)

