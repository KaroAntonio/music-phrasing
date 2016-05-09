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

#sr = None disables resampling
y, sr_x = librosa.load('assets/verite.wav',  sr=None, duration=10.0)
x, sr_y = librosa.load('assets/bangn.wav',  sr=None, duration=10.0)

#THE KEY FUNCTION of seperation *******
y_harmonic, y_percussive = librosa.effects.hpss(y)

#beats
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr_y, trim=False)

#CLICK TRACK
y_beats = librosa.clicks(frames=beats, sr=sr_y)

mix = np.array([(x + y)/2 for x, y in zip(y_beats, y)],dtype=np.float32)

#input array must be in numpy.float32!
librosa.output.write_wav('assets/mix.wav', mix, sr_y)

