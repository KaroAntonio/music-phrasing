#https://sites.google.com/site/haskell102/home/frequency-analysis-of-audio-file-with-python-numpy-scipy

from matplotlib.pyplot import plot, show, title, xlabel, ylabel, subplot, savefig
#from  import plot, figure
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
import numpy

def plotSpectru(y,Fs):
    n = len(y) # lungime semnal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range
    
    threshold = 200;
    Y = fft(y)/n # fft computing and normalization
    Y = abs(Y)
    Y = Y[range(n/2)]
    
    #Y = numpy.array(filter(lambda x: x >= threshold, Y))
    
    frq_slice = 20
    Y = Y[:Y.size/frq_slice]
    frq = frq[:frq.size/frq_slice]
    
    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')

Fs = 44100;  # sampling rate

rate,data=read('assets/verite.wav')
y=data[:,1]
lungime=len(y)
timp=len(y)/rate
t=linspace(0,timp,len(y))

#subplot(2,1,1)
#plot(t,y)
#xlabel('Time')
#ylabel('Amplitude')
#subplot(2,1,2)
plotSpectru(y,Fs)
show()
