from brian2 import *

K_VALUE = 2
M = 28
N = 28
DIGIT_DURATION = 200 * ms
MAX_NUM_NEURONS = 2# Max number of output neurons. Used in finding error

#STDP Parameters
taupre = 34 * ms;
taupost = 14 * ms
apre = 103e-2
apost = -40

<<<<<<< HEAD

NumOfDigitsTrain = 200
=======
NumOfDigitsTrain = 500
>>>>>>> d4419981fbeccaa28eb70fdc0e90bf9a80292706
NumOfDigitsTest = 10

NUM_OUTPUT_CLASSES = 2
DIGITS = [4
    , 1]

timeInh = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh.append(2) #number of time samples should be NUM_OUTPUT_CLASSES

timeExh = [2] #only one time sample

Syn12Condition = 'I += 50 * volt/second'
