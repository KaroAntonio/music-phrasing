#methods for algorithm evaluation

from analysis.phrases import *
from analysis.util import *

import numpy as np
import sys
import json

# Librosa for audio
import librosa

def test_phrase_detect(phrases_path, sample_duration, window):
    #param: 
    #   ids = an array of file_ids of tracks
    #   phrases_path = the filepath for a json object of phrase intervals
    #   sample_duration = duration of tracks to be tested on
    #return: some scalar reflecting how 'close' phrase detection is to the given benchmark
    
    with open(phrases_path) as data_file:    
        benchmarks = json.load(data_file)
        
    ids = benchmarks.keys()
        
    score = 0
    for file_id in ids:
        #LOAD WAVEFORM
        audio_path = 'assets/'+file_id+'.wav'
        y, sr = (librosa.load(audio_path,  sr=None,  duration=sample_duration))
        
        bench = []
        for p in benchmarks[file_id]:
            if p <= sample_duration:
                bench.append(p)
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

        detected = get_phrase_intervals(file_id,y,sr, 1.0, 1.0, 4.0, 20, 0.13, tempo)
        s = evaluate_phrases(bench, detected, window)
        print(file_id +" score: " + str(s))
        score += s
    return score/len(ids)
        
def evaluate_phrases(benchmark, detected, window):
    #return score based on how well the detected phrases match up to the benchmark\
    #hits / maxHits - misses / maxMisses
    
    #convert to frames
    benchmark = librosa.time_to_frames(benchmark,hop_length=256)
    
    print(benchmark)
    print(detected)
    
    hits = 0
    for i in range(0, len(benchmark)):
        target = benchmark[i]
        h_i = hitIndex(target, detected, window)
        if (h_i != -1):
            hits += 1
            
    max_hits = len(benchmark)  
    max_misses = len(benchmark) + len(detected)
    misses = max_misses - (hits*2)
    
    return (hits/max_hits) - (misses/max_misses)
    

def hitIndex(target, phrases, window):
    for i in range(0, len(phrases)):
        p = phrases[i]
        if abs(target-p) < window:
            return i
    return -1
        
        
    
    
        
        
        
    