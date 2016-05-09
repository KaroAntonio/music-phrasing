import librosa
import numpy as np
import scipy as sc
import sys, json, codecs, math
from math import sqrt

def load_array(file_path):
    #load s-matrix from a json file
    print('Loading \''+file_path+'\'...')
    with open(file_path) as json_data:
        a = np.array(json.load(json_data))
    return a

def save_array(a, file_path):
    #saves s-matrix to a json file
    #param:
    #a = np.array
    #file_path = 'storage/'+file_id + '.json'
    with open(file_path, 'w') as json_file:
        json.dump(a.tolist(), json_file)
