from brian2 import *

K_VALUE = 2
M = 28
N = 28
DIGIT_DURATION = 200 * ms
MAX_NUM_NEURONS = 10# Max number of output neurons. Used in finding error

#STDP Parameters
taupre = 20*ms; taupost = taupre
apre = 1.0e-12; apost = apre * taupre / taupost * 1.0
alpha = 0.01

NumOfDigitsTrain = 250
NumOfDigitsTest = 10

NUM_OUTPUT_CLASSES = 10
DIGITS = [0,1,2,3,4,5,6,7,8,9]

timeInh = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh.append(2) #number of time samples should be NUM_OUTPUT_CLASSES

timeExh = [2] #only one time sample

Syn12Condition = 'isyn += 50 * nA'
