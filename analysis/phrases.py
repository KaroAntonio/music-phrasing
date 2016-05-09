#Here be methods for phrase detection and synth
import librosa
import numpy as np
import scipy as sc
import sys, json, codecs, math
from math import sqrt
from analysis.util import *
        
def get_phrase_intervals(file_id, y,sr, r, w, w_p_ratio,  period_threshold, peak_window, tempo):
    #param:
    #   y = waveform
    #   sr = sample rate
    #   r = radius/resolution of diagonal cut
    #   w = checkerboard window size in seconds,  w<=r
    #   w_p_ratio = ratio used for peak picking, unknown
    #   period_threshold = threshold for filtering by period
    #   fpb = frames per beat for phrase search
    
    #TEST CONSTANTS
    if (w>r):
        sys.exit('Window Resolution Mismatch')
    sample_length = librosa.samples_to_time([len(y)],sr)[0] #s
    w_f = librosa.time_to_frames([w],hop_length=256)[0]
    f  = extract_features(y)
    #frames per beat
    fpb = librosa.time_to_frames([1/(tempo/60)],hop_length=256)[0]
    #LOAD OR CREATE S-MATRIX & NOVELTY VECTOR
    s_matrix = init_smatrix(file_id,f,r,  sample_length)
    novelty = init_novelty_vector(file_id, w, w_f,  sample_length, s_matrix)
    
    w_p = w_f/w_p_ratio
    peaks = librosa.util.peak_pick(novelty, w_p, w_p, w_p, w_p, peak_window, w_p)
    
    return filter_by_period(peaks, period_threshold, fpb)
    
def init_smatrix(file_id, f, r, sample_duration, tag=""):
    #param: matrix_id = string unique id of a matrix file
    #   load or create matrix
    #   if matrix must be computed, save immediately
    #return: loaded or computed matrix
    
    matrix_id = file_id + '_' + str(sample_duration) + '_' + str(r) 
    if tag != "":
        matrix_id = matrix_id + '_' + tag
    try:
        m = load_array('storage/'+matrix_id+ '.json');
    except:
        m = get_smatrix_diagonal(f,r);
        save_array(m,'storage/'+matrix_id+'.json');
        
    return m
    
def get_smatrix(f):
    #param: feature array, 
    #compare f to itself
    print('Computing S-Matrix')
    
    dim = len(f[0])
    matrix = np.zeros([dim, dim])
    i_max = int(dim)
    
    for i in range(0,i_max):
        sys.stdout.write("\r" + str(int(i/i_max*100)) + '%') 
        sys.stdout.flush()
        for j in range(0,dim):
            matrix[i][j] = feature_distance(f[:,i], f[:,j])
    sys.stdout.write("\n") 
    return matrix

def init_novelty_vector(file_id, w, w_f, sample_duration, s_matrix, tag=""):
    novelty_id = file_id + '_n_' + str(sample_duration) + '_' + str(w) 
    if tag != "":
        novelty_id = novelty_id + '_' + tag
    try:
        novelty = load_array('storage/'+novelty_id+ '.json')
    except:
        c_matrix = create_c_matrix(w_f)
        novelty = get_novelty_vector(s_matrix, c_matrix)
        save_array(novelty,'storage/'+novelty_id+ '.json');
        
    return novelty
    

def get_novelty_vector(s_matrix, c_matrix):
    #param:
    #s_matrix = similiarity matrix
    #c_matrix = checkerboard matrix
    #return a vector of novelty scores corresponding to the frames of the s_matrix
    
    n = np.zeros([len(s_matrix)])
    print('Calculating Novelty...')
    for i in range(0, len(s_matrix)):
        sys.stdout.write("\r" + str(int((i+1)/len(s_matrix)*100)) + '%') 
        sys.stdout.flush()
        n[i] = get_novelty_score(s_matrix, c_matrix, i,i)
    sys.stdout.write("\n") 
    #get average, discluding entries with -1
    sum = 0
    count = 0
    for i in range(0, len(n)):
        if (n[i] != -1):
            sum += n[i]
            count +=1
    ave = sum / count
    
    #get average dist, discluding entries with -1
    sum = 0
    count = 0
    for i in range(0, len(n)):
        if (n[i] != -1):
            sum += n[i]-ave
            count +=1
    ave_dist = abs(sum/count)
    
    #adjust novelty as distance from average
    n_d = np.zeros([len(n)])
    for i in range(0, len(n)):
        if (n[i] != -1):
            n_d[i] = abs(n[i]-ave)/ave_dist #VERIFY
        else:
            n_d[i] = 0
    
    #scale n_d to range[0,1]
    o_min = min(n_d)
    OldRange = (max(n_d) - min(n_d))  
    for i in range(0, len(n_d)):
         n_d[i] = (((n_d[i] - o_min) * 1) / OldRange)
    
    return n_d

def get_novelty_score(s_matrix, c_matrix, i_m,j_m):
    #param: 
    #c_matrix =  the checkerboard matrix used to correlate to the points of the S-Matrix
    #i,j = row and col of the point to find the novelty of
    #s-matrix with which to correlate
    #find average correlation between two matrices c = 1-abs(a1-b1)
    
    d = len(c_matrix)
    d2 = math.floor(d/2)
    if ((i_m-d2 < 0) or (j_m-d2 < 0)):
        return -1
    
    #cutout submatrix from smatrix
    #rasterize submatrixes (matrix.flatten)
    #cosine distance between s and c
    
    sum_cor = 0
    for i in range(0,d):
        for j in range(0,d):
            j_1 = j-d2+j_m
            i_1 = i-d2+i_m
            try:
                #VERIFY
                cor= 1-abs(c_matrix[i-d2][j-d2]-s_matrix[i_1][j_1])
            except: 
                return -1
            sum_cor += cor
            
            
    return sum_cor/(d*d)

def create_c_matrix(d):
    #param: d=(float) dimension of c_matrix
    #create a checkerboard matrix of size d
    #ex 2x2 c_matrix
    #0011
    #0011
    #1100
    #1100
    
    cm = np.zeros([d, d])
    for i in range (0,2):
        for j in range (0,int(d/2)):
            for k in range (0,int(d/2)):
                cm[j + d/2*i][k + d/2*i] = 1
                
    return cm

def cross_reference(a1, a2, threshold):
    #param: a1, a2 = np.arrays of events, measured in frames
    #       threshold= max variation between events (float)
    #return: an np.array containing only the events that are within threshold of each other
    a3 = []
    for e1 in a1:
        for e2 in a2:
            if (abs(e1-e2) < threshold):
                a3.append(e1)
                break
    return np.array(a3)

def filter_by_period(a, threshold, fpb):
    #param: a = an np.array of events
    #       threshold= max variation between events (float)
    #return: a with non-periodic events filtered out
    #TODO: currently assumes the period is the same for the entire song (not true)
    
    #find potential periods
    sequential_distances = get_sequential_distances(a)
        
    #get cluster means
    #... in other words, the mean potential periods
    cluster_means = get_cluster_means(sequential_distances, threshold)
    
    #find best fitting periodical array
    best_periodical = get_best_periodical(a, cluster_means, threshold, fpb)
    
    #return cross_reference(a, best_periodical, threshold)
    return best_periodical

def get_best_periodical(a, periods, threshold, fpb):
    #param: a = an np.array of values
    #       threshold= max variation between values (float)
    #       periods = a list of potential periods
    #       fpb = frames per beat
    #return: the best fitting periodical array
    
    #ADD 4,8 bar periods into the mix
    periods = periods + [fpb*16,fpb*32]
    
    best_score = float("-inf")
    best_periodical = []
    #for each period
    for p in periods:
        score, periodical = evaluate_period(a, p, threshold, fpb)
        if score > best_score:
            best_score = score
            best_periodical = periodical
        
    return best_periodical 
        
def evaluate_period(a, period, threshold, fpb):
    #param: a = an np.array of values
    #       threshold= max variation between values (float)
    #       period = a potential period
    #return: the best score, and periodical array of a period 
    
    #determine how well period fits a
    best_score = float("-inf")
    best_periodical = []
    num_intervals = int((a[-1]-a[0])/period)
    for i in range(0, len(a)):
        periodical = [a[i]]
        hits = 0
        #compute score for forward half of a
        for j in range(i, num_intervals-1):
            target = periodical[j-i] + period
            v = find_value(a, target, threshold)
            if v != -1:
                periodical.append(v)
                hits += 1
            else:
                periodical.append(target)
        
        #compute score for backward half of a
        for j in range(0, i):
            target = periodical[0] - period
            if target > 0:
                v = find_value(a, target, threshold)
                if v != -1:
                    periodical.insert(0,v)
                    hits += 1
                else:
                    periodical.insert(0,target)
        
        score = hits/len(periodical)
        
        #BIAS periods close to 4,8 Bars in length
        
        if (abs(period-fpb*16) < (threshold*16)):
            score = score**(1/4)
        if (abs(period-fpb*16) < (threshold*16)):
            score = score**(1/4)
            
        if score > best_score:
            best_score = score
            best_periodical = periodical
            
    return best_score, best_periodical
    
def find_value(a, target, threshold):
    #param: a = an np.array of values
    #       threshold= max variation between values (float)
    #return:    The first value found within threshold of target (terrible...)
    #           -1 if no value is found
    
    for e in a:
        if abs(e-target) <= threshold:
            return e
    return -1
    
def get_cluster_means(a, threshold):
    #param: a = an np.array of values
    #       threshold= max variation between values (float)
    #return: list of the mean value of data clusters
    #TODO: this is probably formalized somewhere, what's the best technique?
    
    cluster_means = []
    for cluster in parse_clusters(np.sort(a), threshold):
        mean, stdev = stat(cluster)
        cluster_means.append(mean)
        
    return cluster_means; 
    
def stat(lst):
    #http://stackoverflow.com/questions/8940049/how-would-you-group-cluster-these-three-areas-in-arrays-in-python
    """Calculate mean and std deviation from the input list."""
    n = float(len(lst))
    mean = sum(lst) / n
    stdev = sqrt((sum(x*x for x in lst) / n) - (mean * mean)) 
    return mean, stdev

def parse_clusters(lst, n):
    #http://stackoverflow.com/questions/8940049/how-would-you-group-cluster-these-three-areas-in-arrays-in-python
    #param: a sorted list-like of values s.t. lst[0]<=lst[-1]
    
    cluster = []
    for i in lst:
        if len(cluster) <= 1:    # the first two values are going directly in
            cluster.append(i)
            continue

        mean,stdev = stat(cluster)
        if abs(mean - i) > n:    # check the "distance"
            yield cluster
            cluster[:] = []    # reset cluster to the empty list

        cluster.append(i)
    yield cluster           # yield the last cluster

def get_sequential_distances(a):
    #param: a = an np.array of values
    #return: np.array of the distances between sequential values
    sequential_distances = []
    for i in range(0,len(a)-1):
        d = a[i+1]-a[i]
        if (d >= 0):
            sequential_distances.append(d)
    return np.array(sequential_distances);
    
def get_smatrix_diagonal(f, r):
    #param: f=feature array, r=radius/resolution of of matrix sampled from the diagonal in seconds
    #optimized based on the assumption that only information along the diagonal of the matrix is important
    print('Computing S-Matrix Diagonal')
    
    #convert r in seconds to r in frames
    r_f = librosa.time_to_frames(np.array([r]), sr=22050, hop_length=256)[0]
    
    dim = len(f[0])
    if (r_f > dim):
        r_f = -dim
    matrix = np.zeros([dim, dim])
    i_max = int(dim)
    
    for i in range(0,dim):
        sys.stdout.write("\r" + str(int((i+1)/dim*100)) + '%') 
        sys.stdout.flush()
        for j in range(0,r_f*2):
            i_r = i
            j_r = max(min(j + i-r_f,dim-1),0)
            matrix[i_r][j_r] = feature_distance(f[:,i_r], f[:,j_r])
    sys.stdout.write("\n") 
    return matrix
            
def extract_features(wave_form):
    #exract arbitrary sound 'features'
    #return feature set
    
    return  np.abs(librosa.stft(wave_form))

def feature_distance(v1, v2):
    #measure the distance between two feature vectors
    
    return 1-sc.spatial.distance.cosine(v1, v2)
