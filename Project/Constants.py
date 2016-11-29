from brian2 import *

K_VALUE = 2
M = 28
N = 28
DIGIT_DURATION = 200 * ms
MAX_NUM_NEURONS = 10# Max number of output neurons. Used in finding error

#STDP Parameters
taupre = 34 * ms;
taupost = 14 * ms
apre = 103e-2
apost = -40

NumOfDigitsTrain = 500
NumOfDigitsTest = 10

NUM_OUTPUT_CLASSES = 10
DIGITS = [0, 4, 6, 2, 3, 5, 1, 7, 8, 9]

Syn12Condition = 'I += 50 * volt/second'