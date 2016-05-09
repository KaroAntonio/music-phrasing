import numpy as np
import sys

# Librosa for audio
import librosa

# matplotlib for displaying the output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#matplotlib inline

# And seaborn to make it look nice
#import seaborn
#seaborn.set(style='ticks')

# and IPython.display for audio output
#import IPython.display

from analysis.phrases import *
from analysis.util import *

#test to attempt to overlap the beggining of one song (y) with the end of another (x)
            
#CONSTANTS
x_file_id = 'loud'
y_file_id = 'obvs'
sample_duration = 30.0   
r = 1.0     #r = radius/resolution of diagonal cut
w = r       #w = checkerboard window size in seconds,  w<=r
w_p_ratio = 4
#WHERE DO THESE THRESHOLD VALUES COME FROM?
beat_threshold = 15
period_threshold = 20
peak_window = 0.13

#TEST CONSTANTS
if (w>r):
    sys.exit('Window Resolution Mismatch')
     
#LOAD WAVEFORMS
x_audio_path = 'assets/'+x_file_id+'.wav'
y_audio_path = 'assets/'+y_file_id+'.wav'
x, x_sr = (librosa.load(x_audio_path,  sr=None))
y, y_sr = (librosa.load(y_audio_path,  sr=None,duration=sample_duration))

#Get Beats
y_harmonic, y_percussive = librosa.effects.hpss(y)
x_harmonic, x_percussive = librosa.effects.hpss(x)
y_tempo, y_beats = librosa.beat.beat_track(y=y_percussive, sr=y_sr)
x_tempo, x_beats = librosa.beat.beat_track(y=x_percussive, sr=x_sr)

#Match Tempos to 120
y = librosa.effects.time_stretch(y, 120/y_tempo)
x = librosa.effects.time_stretch(x, 120/x_tempo)

#Get Beats
y_harmonic, y_percussive = librosa.effects.hpss(y)
x_harmonic, x_percussive = librosa.effects.hpss(x)
y_tempo, y_beats = librosa.beat.beat_track(y=y_percussive, sr=y_sr)
x_tempo, x_beats = librosa.beat.beat_track(y=x_percussive, sr=x_sr)

#Analyze end of x
x_s = librosa.time_to_samples([sample_duration], sr=x_sr)
x = x[len(x)-x_s:-1]

w_f = librosa.time_to_frames([w],hop_length=256)[0]
w_p = w_f/w_p_ratio
y_f  = extract_features(y)
x_f  = extract_features(x)

y_fpb = librosa.time_to_frames([1/(y_tempo/60)],hop_length=256)[0]
x_fpb = librosa.time_to_frames([1/(x_tempo/60)],hop_length=256)[0]

x_s_matrix = init_smatrix(x_file_id,x_f,r, sample_duration, tag='end_120bpm')
y_s_matrix = init_smatrix(y_file_id,y_f,r, sample_duration,tag='120bpm')

x_novelty = init_novelty_vector(x_file_id, w, w_f, sample_duration, x_s_matrix, tag='end_120bpm')
y_novelty = init_novelty_vector(y_file_id, w, w_f, sample_duration, y_s_matrix,tag='120bpm')

x_peaks = librosa.util.peak_pick(x_novelty, w_p, w_p, w_p, w_p, peak_window, w_p)
y_peaks = librosa.util.peak_pick(y_novelty, w_p, w_p, w_p, w_p, peak_window, w_p)

x_peaks = filter_by_period(x_peaks, period_threshold,x_fpb)
y_peaks = filter_by_period(y_peaks, period_threshold,y_fpb)

x_p_s = librosa.frames_to_samples(x_peaks)
y_p_s = librosa.frames_to_samples(y_peaks)

sample_x = x[x_p_s[0]:x_p_s[1]]
sample_y = y[y_p_s[0]:y_p_s[1]]

hard_cut = np.concatenate((sample_x,sample_y))

librosa.output.write_wav('mixes/hard_cut.wav', hard_cut, x_sr)

