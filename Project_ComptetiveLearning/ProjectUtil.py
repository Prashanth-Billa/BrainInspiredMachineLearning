import numpy as np
import matplotlib.pyplot as plt
import matplotlib,sys
import os, struct
from array import array as pyarray
from numpy import array, int8, uint8, zeros
from brian2 import *
from Constants import *

def getIndicesInh(label):
    arr = []
    label_index = DIGITS.index(label)
    for i in range(0,NUM_OUTPUT_CLASSES):
        if label_index == i:
            continue
        arr.append(i)
    return arr

def getStateMonitor(layer):
    monitor = {}
    monitor['voltage'] = StateMonitor(layer, variables='v', record=[0])
    monitor['current'] = StateMonitor(layer, variables='isyn', record=[0])
    return monitor

def getSpikeMonitor(layer):
    return SpikeMonitor(layer)

def getError(spikeMonitorObject, targetLabels, phase=0):
    map = {}
    NumOfDigits = 0
    if(phase == 0):
        NumOfDigits = NumOfDigitsTrain
    else:
        NumOfDigits = NumOfDigitsTest

    for index in range(0, NumOfDigits):
        map[index] = []

    for index in range(0, len(spikeMonitorObject.i)):
        key = int((spikeMonitorObject.t / ms)[index] / (DIGIT_DURATION / ms + 0.0001))
        if key in map:
            arr = map[key]
            arr.append(spikeMonitorObject.i[index])
        else:
            arr = []
            arr.append(spikeMonitorObject.i[index])
            map[key] = arr

    err = 0
    total = NumOfDigits

    for key, value in map.iteritems():
        arr = []
        a, b, c, d = getMaxfreqNeuronIndex(value)
        print "Prediction :{0}, Target: {1}".format(DIGITS[a], targetLabels[key])
        if DIGITS[a] != targetLabels[key]:
            err = err + 1
        map[key] = arr

    return float(err)/total

def getMaxfreqNeuronIndex(indices):
    countarr = {}
    #supports upto 100 neurons
    for i in range(0, len(indices)):
        key = indices[i]
        if key in countarr:
            v = countarr[indices[i]]
            v = v + 1
            countarr[indices[i]] = v
        else:
            countarr[indices[i]] = 1

    maxVal = -1
    index = -1

    nextMaxVal = -1
    nextIndex = -1
    for i in range(0, MAX_NUM_NEURONS):
        if i in countarr:
            if countarr[i] > maxVal:
                nextMaxVal = maxVal
                nextIndex = index
                maxVal = countarr[i]
                index = i
            elif countarr[i] > nextMaxVal:
                nextMaxVal = countarr[i]
                nextIndex = i

    return index, maxVal, nextIndex, nextMaxVal

def load_mnist_60000(dataset="training", digits=np.arange(10), path="."):
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
    url_data = "mnist_data.npy"
    url_targets = "mnist_targets.npy"
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