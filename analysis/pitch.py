#Here be methods for phrase detection and synth
import librosa
import numpy as np
import scipy as sc
import sys, json, codecs, math
from math import sqrt
from analysis.util import *

def get_pitch_sums(chroma):
    #param: chroma = a chromagram catagorized into 12 western-scale pitches
    #return: normalized pitch classes
    #       C ,C#,D ,D#,E ,F ,F#,G ,G#,A ,A#,B
    #       0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11
    
    pitch_sums = []
    for pitch_class in chroma:
        pitch_sums.append(sum(pitch_class))
        
    return np.array(pitch_sums)

def pitch_correlation(v1, v2):
    
    #param: v1, v2 = arrays of chroma sums for wave slices
    #return: a float of the correlation between the two waves
    return 1-sc.spatial.distance.cosine(v1, v2)
    
    
    
    

        