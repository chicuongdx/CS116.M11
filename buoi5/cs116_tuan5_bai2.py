import os
from scipy.io import wavfile
from IPython.display import Audio
import numpy as np
import librosa
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

file = "musicFile.wav"
x, fs = librosa.load(file)
Audio(x, rate=fs)
X = librosa.feature.mfcc(x, sr=fs)
X = preprocessing.scale(X)
model = PCA(n_components=2, whiten=True)
model.fit(X.T)
Y = model.transform(X.T)
print(Y.shape)
plt.scatter(Y[:,0], Y[:,1])


'''def transfer_PCA(color_layer):
    pca = PCA(n_components=50)
    pca.fit(color_layer)
    trans_pca = pca.transform(color_layer)
    return pca, trans_pca
#listen to music
file = "musicFile.wav"
file_name = os.path.dirname(os.path.abspath(file)) + "/" + file

fs, wav = wavfile.read(filename=file)
wav_rs = wav.reshape(512, int((wav.shape[0] * wav.shape[1])/512))

rever_mus = []
for row in wav_rs:
    pca, trans = transfer_PCA(row)
    ms_arr = pca.inverse_transform(trans)
    rever_mus.append(ms_arr)

rever_wav = np.array(rever_mus)
print(rever_wav)'''