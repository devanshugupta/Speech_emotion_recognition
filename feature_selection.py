import numpy as np
import librosa as lr
import librosa.display
from IPython.display import Audio
import os
import csv
header = 'filename rmse chroma mel tonnetz'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('H:/data4.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

actors = []
for _ in range(1, 10):
    actors.append('Actor_0' + str(_))
for _ in range(10, 25):
    actors.append('Actor_' + str(_))

for a in actors:
    for filename in os.listdir(f'H:/Audio_Song_Actors/{a}'):
        audio = f'H:/Audio_Song_Actors/{a}/{filename}'
        y, sr = lr.load(audio)

        d={'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}

        rmse = librosa.feature.rms(y=y)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        mel = np.mean(librosa.feature.melspectrogram(y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))

        to_append = f'{filename} {np.mean(rmse)} {chroma} {mel} {tonnetz}'
        for i in mfcc:
            to_append += f' {np.mean(i)}'
        s=filename.split('-') #Labels

        to_append += f' {d[s[2]]}'
        #data4 : Song_Actors
        #data3 : Speech_Actors
        file = open('H:/data4.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
