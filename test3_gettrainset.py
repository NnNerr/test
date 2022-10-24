from scipy.io import wavfile
import os
import numpy as np
import utils
import random

ID_nums ={1:52,2:45,3:11,4:33,5:28,6:42,7:37,8:45,9:26,10:43
         ,11:12,12:21,13:22,14:19,15:34,16:31,17:7,18:16,19:24,20:17}

for count in range(20):
    traindata_num = 1
    for file_name in os.listdir(f'./train/ID{count+1}'):
        audio_trace0 = utils.read_audio(os.path.join(f'./train/ID{count+1}',file_name),sr=44100)
        id1 = random.randint(1,20)
        id2 = random.randint(1,20)
        while id1 == count+1 or id2 == count+1 or id1 == id2:
            id1 = random.randint(1,20)
            id2 = random.randint(1,20)
        
        num1 = random.randint(1,ID_nums[id1])
        num1 = random.randint(1,ID_nums[id2])

        j=1
        for file_name1 in os.listdir(f'./train/ID{id1}'):
            if j == num1:
                audio_trace1 = utils.read_audio(os.path.join(f'./train/ID{id1}',file_name1),sr=44100)
                break
            j += 1
        j=1
        for file_name2 in os.listdir(f'./train/ID{id2}'):
            if j == num1:
                audio_trace2 = utils.read_audio(os.path.join(f'./train/ID{id2}',file_name2),sr=44100)
                break
            j += 1
        

        minlength = min(len(audio_trace0),len(audio_trace1),len(audio_trace2))

        audio_trace0 = audio_trace0[:minlength]
        audio_trace1 = audio_trace1[:minlength]
        audio_trace2 = audio_trace2[:minlength]

        mix_audio = audio_trace0 + audio_trace1 + audio_trace2

        wavfile.write(f'./train3/ID{count+1}/clean_audio{traindata_num}.wav',44100,audio_trace0)
        wavfile.write(f'./train3/ID{count+1}/mix_audio{traindata_num}.wav',44100,mix_audio)

        if traindata_num == 10:
            break
        traindata_num += 1

