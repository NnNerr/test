import os
import numpy as np
import utils
import pickle
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc
from python_speech_features import delta



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

for i in range(20):
    features = np.array(())
    for file_name in os.listdir(f'./train/ID{i+1}'):
        audio_trace1 = utils.read_audio(os.path.join(f'./train/ID{i+1}',file_name),sr=16000)
        vector1 = extract_features(audio_trace1,16000)
        #B1 = B.transpose(1,0).reshape(-1, 1)
        if features.size == 0:
            features = vector1
        else:
            features = np.vstack((features, vector1))
    gmm = GaussianMixture(16, covariance_type='full',max_iter=150,n_init = 3)
    gmm.fit(features)
    picklefile = f'{i+1}.gmm'
    with open(picklefile, 'wb') as gmm_file:
        pickle.dump(gmm, gmm_file)
    print(i)


