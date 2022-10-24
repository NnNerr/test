import os
import numpy as np
import utils
from scipy.io import wavfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import soundfile as sf


def seperator(ID, mix1):
    model1 = load_model(f'./task3_model/model{ID}.h5')

    win_length = 512
    hop_length = 128
    nfft = 512
    spectrum = librosa.stft(mix1, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    magnitude = np.abs(spectrum).T
    phase = np.angle(spectrum).T

    frame_num = magnitude.shape[0] - 4
    feature = np.zeros([frame_num, 257 * 5])
    k = 0
    for i in range(frame_num - 4):
        frame = magnitude[k:k + 5]
        feature[i] = np.reshape(frame, 257 * 5)
        k += 1

    ss = StandardScaler()
    feature = ss.fit_transform(feature)
    feature = np.expand_dims(feature, axis=1)
    mask = model1.predict(feature)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    mask1 = np.zeros([mask.shape[0], 257])
    for i in range(mask.shape[0]):
        mask1[i] = np.reshape(mask[i], 257)

    magnitude = magnitude[2:-2]
    en_magnitude = np.multiply(magnitude, mask1)
    phase = phase[2:-2]

    en_spectrum = en_magnitude.T * np.exp(1.0j * phase.T)
    rev_audio1 = librosa.istft(en_spectrum, win_length=win_length, hop_length=hop_length)

    return np.float32(rev_audio1)


if __name__=='__main__':
    audio_trace,_ = librosa.load(os.path.join('./train3/ID20','mix_audio3.wav'),sr=44100)
    #audio_trace,_ = librosa.load(os.path.join('./test_offline/task3','combine003.mp4'),sr=16000)
    audio = seperator(20,audio_trace)
    sf.write('_left.wav',  audio, 44100)
    print()
