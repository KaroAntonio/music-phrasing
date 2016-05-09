import numpy as np
import sys
import json

# Librosa for audio
import librosa

# matplotlib for displaying the output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#matplotlib inline

# And seaborn to make it look nice
import seaborn
seaborn.set(style='ticks')

# and IPython.display for audio output
import IPython.display

from analysis.phrases import *
from analysis.util import *
            
#CONSTANTS
file_id = 'all'
sample_duration = 20.0   
r = 1.0     #r = radius/resolution of diagonal cut
w = r       #w = checkerboard window size in seconds,  w<=r
w_p_ratio = 4
#WHERE DO THESE THRESHOLD VALUES COME FROM?
beat_threshold = 15
period_threshold = 20
peak_window = 0.13

#BENCHMARK
with open('assets/phrase_intervals.json') as data_file:    
    data = json.load(data_file)
    
bench = []
for p in data[file_id]:
    if p <= sample_duration:
        bench.append(p)
        
bench = librosa.time_to_frames(bench,hop_length=256)

#TEST CONSTANTS
if (w>r):
    sys.exit('Window Resolution Mismatch')
     
#LOAD WAVEFORM
audio_path = 'assets/'+file_id+'.wav'
y, sr = (librosa.load(audio_path,  sr=None,  duration=sample_duration))

w_f = librosa.time_to_frames([w],hop_length=256)[0]
f  = extract_features(y)

#Get Beats
y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
fpb = librosa.time_to_frames([1/(tempo/60)],hop_length=256)[0]

#LOAD OR CREATE S-MATRIX & NOVELTY VECTOR
s_matrix = init_smatrix(file_id,f,r, sample_duration)
novelty = init_novelty_vector(file_id, w, w_f, sample_duration, s_matrix)

#https://bmcfee.github.io/librosa/generated/librosa.util.peak_pick.html?
#TODO correlate to the beat, somehow
w_p = w_f/w_p_ratio
peaks = librosa.util.peak_pick(novelty, w_p, w_p, w_p, w_p, peak_window, w_p)

#cross reference beats and peaks
#peaks = cross_reference(beats, peaks, beat_threshold)
#assuming music is periodic...
peaks = filter_by_period(peaks, period_threshold,fpb)

#Sample a test segment
p_s = librosa.frames_to_samples(peaks)
if (len(p_s) > 2):
    sample = y[p_s[1]:p_s[2]]
    librosa.output.write_wav('mixes/sampled.wav', sample, sr)
    loop = np.concatenate([sample,sample,sample])
    librosa.output.write_wav('mixes/loop.wav', loop, sr)
    
#Shuffle a test segment
p_s = librosa.frames_to_samples(peaks)
if (len(p_s) >= 4):
    s1 = y[p_s[0]:p_s[1]]
    s2 = y[p_s[1]:p_s[2]]
    s3 = y[p_s[2]:p_s[3]]
    loop = np.concatenate([s3,s2,s1])
    librosa.output.write_wav('mixes/shuffle.wav', loop, sr)
    
    #layer phrases
    layer = np.array([(x + y)/2 for x, y in zip(s1, s2)],dtype=np.float32)
    layer = np.array([(x + y)/2 for x, y in zip(layer, s3)],dtype=np.float32)
    librosa.output.write_wav('mixes/layered.wav', layer, sr)
   
#GRAPHING
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(5,4)
ax1 = fig.add_subplot(gs[0:4,:])
for p in peaks:
    ax1.axvline(p)
librosa.display.specshow(s_matrix, hop_length=256, x_axis='time', y_axis='time', aspect='equal')
ax1.vlines(bench, 0, s_matrix.shape[0], colors='r', linestyles='-', linewidth=2, alpha=0.5)
plt.title(file_id + ' STFT distance (symmetric)')
ax2 = fig.add_subplot(gs[4,:])
ax2.plot(novelty)
#ax2.vlines(peaks, 0, novelty.shape[0], colors='b', linestyles='-', linewidth=2, alpha=0.5)
#for p in peaks:
#    ax2.axvline(p)
plt.show()
