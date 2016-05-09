#Match one track up to the beats of another track
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
#mono=True, loads in a mono waveform
x, sr_x = librosa.load('assets/all.wav',  sr=None, duration=20.0, mono=True)
y, sr_y = librosa.load('assets/hazey.wav',  sr=None, duration=20.0, mono=True)

#THE KEY FUNCTION of seperation *******
y_harmonic, y_percussive = librosa.effects.hpss(y)
x_harmonic, x_percussive = librosa.effects.hpss(x)

#beats
tempo_y, beats_y = librosa.beat.beat_track(y=y_percussive, sr=sr_y, trim=True)
tempo_x, beats_x = librosa.beat.beat_track(y=x_percussive, sr=sr_x, trim=True)

#adjust x to be the same tempo as y
ym = librosa.effects.time_stretch(y, tempo_x/tempo_y)

#remeasure tempo of y_matched
ym_harmonic, ym_percussive = librosa.effects.hpss(ym)
tempo_ym, beats_ym = librosa.beat.beat_track(y=ym_percussive, sr=sr_y, trim=True)

#PHASE the tracks
#get arrays of the sample indices of the beats
beats_i_x = librosa.frames_to_samples(beats_x)
beats_i_ym = librosa.frames_to_samples(beats_ym)

#cut off the tracks at the beats
x = x[beats_i_x[1]:]
ym = ym[beats_i_ym[1]:]

#mix the matched tracks
mix = np.array([(x + y)/2 for x, y in zip(ym, x)],dtype=np.float32)

#input array must be in numpy.float32!
librosa.output.write_wav('mixes/beat_matched.wav', mix, sr_y)

