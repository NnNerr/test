import os
import numpy as np
import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import librosa
from sklearn.preprocessing import StandardScaler


def IRM(clean_spectrum, mix_spectrum):
    # IRM
    snr = np.divide(np.abs(clean_spectrum), np.abs(mix_spectrum))
    mask = snr / (snr + 1)
    mask[np.isnan(mask)] = 0.5
    mask = np.power(mask, 0.5)
    return mask

def getData(ID):
    data_num = 10
    ss = StandardScaler()
    if ID == 17:
        data_num = 7
    for count in range(data_num):
        clean_wav, _ = librosa.load(os.path.join(f'./train3/ID{ID}',f'clean_audio{count+1}.wav'),sr=44100)
        mix_wav, _ = librosa.load(os.path.join(f'./train3/ID{ID}',f'mix_audio{count+1}.wav'),sr=44100)

        win_length = 512
        hop_length = 128
        nfft = 512

        clean_spectrum = librosa.stft(clean_wav, win_length=win_length, hop_length=hop_length, n_fft=nfft)
        mix_spectrum = librosa.stft(mix_wav, win_length=win_length, hop_length=hop_length, n_fft=nfft)

        clean_mag = np.abs(clean_spectrum.T)
        mix_mag = np.abs(mix_spectrum.T)

        # 5帧一行
        frame_num = mix_mag.shape[0] - 4
        feature1 = np.zeros([frame_num, 257*5])
        k = 0
        for i in range(frame_num):
            frame = mix_mag[k:k+5]
            feature1[i] = np.reshape(frame, 257*5)
            k += 1
        feature1 = ss.fit_transform(feature1)
        mask = IRM(clean_mag, mix_mag)
        label1 = mask[2:-2]

        if count == 0:
            feature = feature1
            label = label1
        else:
            feature = np.vstack((feature,feature1))
            label = np.vstack((label,label1))

    return feature,label


def get_RNNmodel():
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(128, input_dim=1285, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))


    model.add(keras.layers.Dense(257))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    return model

def train(feature, label, model, ID):
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])
    model.fit(feature, label, batch_size=64, epochs=8, validation_split=0.1)
    model.save(f"./task3_model/model{ID}.h5")


for Id in range(20):
    feature, label = getData(Id+1)
    feature = np.expand_dims(feature, axis=1)
    label = np.expand_dims(label, axis=1)
    model = get_RNNmodel()
    train(feature, label, model, Id+1)
