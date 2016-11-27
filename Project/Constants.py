from brian2 import *

K_VALUE = 2
M = 28
N = 28
DIGIT_DURATION = 200 * ms

#STDP Parameters
taupre = 34 * ms;
taupost = 14 * ms
apre = 103e-3
apost = -40

NumOfDigits = 20#data.shape[0]
NUM_OUTPUT_CLASSES = 10
DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Syn12Condition = 'I += 1e-3 * volt/second'
