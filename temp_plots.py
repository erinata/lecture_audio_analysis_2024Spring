import librosa
import matplotlib.pyplot as pyplot
import os

if not os.path.exists("temp_plots"):
  os.makedirs('temp_plots')
  
genre = "pop"
number = "00000"

audio_file = "genres/" + genre + "/" + genre + "." + number + ".wav"
print(audio_file)
x, sr = librosa.load(audio_file, sr=44100)

wave_plot=pyplot.figure(figsize=(13,5))
librosa.display.waveshow(x, sr=sr, color="blue")
pyplot.title("Waveplot of " + genre + "." + number)
pyplot.xlabel("Time")
pyplot.ylabel("Y")
pyplot.savefig("temp_plots/" + genre + "." + number + "_waveplot.png")
pyplot.close()

spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
print(spectral_centroids)
spectral_rolloff = librosa.feature.spectral_rolloff(y=x+0.01, sr=sr)[0]
print(spectral_rolloff)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
print(spectral_bandwidth)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y=x)
print(zero_crossing_rate)
mfcc = librosa.feature.mfcc(y=x, sr=sr)
print(mfcc)


stft_data = librosa.stft(x)
db_data = librosa.amplitude_to_db(abs(stft_data))
spectrogram_plot=pyplot.figure(figsize=(13,5))
librosa.display.specshow(db_data, sr=sr, x_axis="time", y_axis="hz")
pyplot.colorbar()
pyplot.title("Spectrogram of " + genre + "." + number)
pyplot.savefig("temp_plots/" + genre + "." + number + "_spectrogram.png")
pyplot.close()


chroma_stft_data = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=sr)
chromagram = pyplot.figure(figsize=(13,5))
librosa.display.specshow(chroma_stft_data, x_axis="time", y_axis="chroma", hop_length=sr, cmap='coolwarm')
chromagram.savefig("temp_plots/" + genre + "." + number + "_chromagram.png")
pyplot.close()







