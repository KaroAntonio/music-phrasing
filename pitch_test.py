#librosa pitch classes
#librosa harmonic percussive seperation

import numpy as np

# Librosa for audio
import librosa

# matplotlib for displaying the output
import matplotlib.pyplot as plt
#matplotlib inline

# And seaborn to make it look nice
import seaborn
seaborn.set(style='ticks')

# and IPython.display for audio output
import IPython.display

from analysis.phrases import *
from analysis.pitch import *
from analysis.util import *

file_id = 'all'
audio_path = 'assets/'+file_id+'.wav'

#sr = None disables resampling
y, sr = (librosa.load(audio_path,  sr=None,duration=40.0))

#THE KEY FUNCTION of seperation *******
y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, trim=False)

#PHRASE DETECT
phrases = get_phrase_intervals(file_id,y,sr, 1.0 , 1.0, 4.0,  15, 0.13, tempo)
s_phrases = librosa.frames_to_samples(phrases)

y1 = np.array(y[s_phrases[1]:s_phrases[2]])
y2 = np.array(y[s_phrases[3]:s_phrases[4]])

#THE KEY FUNCTION of seperation *******
y_harmonic, y_percussive = librosa.effects.hpss(y1)
y_harmonic_2, y_percussive_2 = librosa.effects.hpss(y2)
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, trim=False)

# We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
# We'll use the harmonic component to avoid pollution from transients
C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
C2 = librosa.feature.chroma_cqt(y=y_harmonic_2, sr=sr)

pitch_sums_1 = get_pitch_sums(C)
pitch_sums_2 = get_pitch_sums(C2)
corr = pitch_correlation(pitch_sums_1 ,pitch_sums_2)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)
exit
#PITCH SHIFTING
#y_up = librosa.effects.pitch_shift(y, sr, n_steps=4)
#librosa.output.write_wav('assets/verite_0_10.wav', y_up, sr)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the chromagram: the energy in each chromatic pitch class as a function of time
# To make sure that the colors span the full range of chroma values, set vmin and vmax
librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
plt.figure(figsize=(12,4))
librosa.display.specshow(C2, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
#plt.vlines(beats, 0, C.shape[0], colors='r', linestyles='-', linewidth=2, alpha=0.5)
#plt.vlines(phrases, 0, C.shape[0], colors='b', linestyles='-', linewidth=2, alpha=0.5)

plt.title('Chromagram + Beats')
plt.colorbar()

plt.tight_layout()
plt.show()
