import pandas
import librosa

import os
import glob

import numpy

genres = ["blues",
          "country",
          "hiphop",
          "metal",
          "reggae",
          "classical",
          "disco",
          "jazz",
          "pop",
          "rock"]
          
for genre in genres:
  print(genre)
  for audio_file in glob.glob("./genres/" + genre + "/*.wav"):
    print(audio_file)
  
    x, sr = librosa.load(audio_file, sr=44100)

    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=x+0.01, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=x)
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    chroma_stft_data = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=sr)
    rms = librosa.feature.rms(y=x)

    new_row = { "audio_file": audio_file,
                "spectral_centroids": numpy.mean(spectral_centroids),
                "spectral_rolloff": numpy.mean(spectral_rolloff),
                "spectral_bandwidth": numpy.mean(spectral_bandwidth),
                "zero_crossing_rate": numpy.mean(zero_crossing_rate),
                "chroma_stft_data": numpy.mean(chroma_stft_data),
                "rms": numpy.mean(rms),
                "mfcc0":numpy.mean(mfcc[0]),
                "mfcc1":numpy.mean(mfcc[1]),
                "mfcc2":numpy.mean(mfcc[2]),
                "mfcc3":numpy.mean(mfcc[3]),
                "mfcc4":numpy.mean(mfcc[4]),
                "mfcc5":numpy.mean(mfcc[5]),
                "mfcc6":numpy.mean(mfcc[6]),
                "mfcc7":numpy.mean(mfcc[7]),
                "mfcc8":numpy.mean(mfcc[8]),
                "mfcc9":numpy.mean(mfcc[9]),
                "mfcc10":numpy.mean(mfcc[10]),
                "mfcc11":numpy.mean(mfcc[11]),
                "mfcc12":numpy.mean(mfcc[12]),
                "mfcc13":numpy.mean(mfcc[13]),
                "mfcc14":numpy.mean(mfcc[14]),
                "mfcc15":numpy.mean(mfcc[15]),
                "mfcc16":numpy.mean(mfcc[16]),
                "mfcc17":numpy.mean(mfcc[17]),
                "mfcc18":numpy.mean(mfcc[18]),
                "mfcc19":numpy.mean(mfcc[19]),
                "genre": genre
                }

    print(new_row)
        





