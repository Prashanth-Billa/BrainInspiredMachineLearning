import numpy as np
import matplotlib.pyplot as plt
import matplotlib,sys
import os, struct
from array import array as pyarray
from numpy import array, int8, uint8, zeros
from brian2 import *

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    if dataset == "training":
        fname_img = 'C:/Users/Prash/Desktop/psych268_bilm-master/code/train-images.idx3-ubyte'
        fname_lbl = 'C:/Users/Prash/Desktop/psych268_bilm-master/code/train-labels.idx1-ubyte'
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def data_load_mnist(digits = None):
    '''
    Download and load MNIST hand-written digits
    Inputs:
    *digits*: list specifying which digits should be returned (default: return all digits 0-9)
    Ouputs:
    *images* a 1000x784np.array storing 1000 28x28 pixel images of hand-written
    digits
    *labels* labels of the 1000 images
    '''
    import os.path
    url_data = "https://drive.google.com/uc?export=download&id=0B7CeL_WOYFxpTl94RHkxN0pfMEk"
    url_targets = "https://drive.google.com/uc?export=download&id=0B7CeL_WOYFxpZmY4T1hYSHJNTjQ"
    fname_data = 'mnist_data.npy'
    fname_targets = 'mnist_targets.npy'
    if sys.version_info[0] == 2:
        if not os.path.isfile(fname_data):
            import urllib
            urllib.urlretrieve(url_data, fname_data)
        if not os.path.isfile(fname_targets):
            import urllib
            urllib.urlretrieve(url_targets, fname_targets)
    elif sys.version_info[0] == 3:
        if not os.path.isfile(fname_data):
            import urllib.request2
            urllib.request2.urlretrieve(url_data, fname_data)
        if not os.path.isfile(fname_targets):
            import urllib.request2
            urllib.request2.urlretrieve(url_targets, fname_targets)
    data = np.load(fname_data)
    labels = np.load(fname_targets)
    if digits is None:
        return data, labels
    else:
        idx = np.zeros_like(labels, dtype = 'bool')
        for d in digits:
            idx+= labels == d
        return data[idx,:], labels[idx]

def getIndicesInh(label):
    arr = []
    for i in range(0, 10):
        if label == i:
            continue
        arr.append(i)
    return arr

def getStateMonitor(layer):
    monitor = {}
    monitor['voltage'] = StateMonitor(layer, variables='v', record=[0])
    monitor['current'] = StateMonitor(layer, variables='I', record=[0])
    return monitor

def getSpikeMonitor(layer):
    return SpikeMonitor(layer)