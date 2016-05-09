import os
from numpy import *
from scipy.fftpack import fft, ifft, ifft
from PIL import Image
from pylab import *
#import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, figure
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy import ndimage

from scipy.io import wavfile

dat = wavfile.read("assets/verite.wav")
w = dat[1][:,0]

w_hat = fft(w)

w_hat_mag = abs(w_hat)
plot(log(1+w_hat_mag)[:20000])
#plot(log(1+w_hat_mag))

#figure(2)

#plot(log(1+w_hat_mag)[:2000])