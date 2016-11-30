from brian2 import *

K_VALUE = 2
M = 28
N = 28
DIGIT_DURATION = 200 * ms
DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_NUM_NEURONS = len(DIGITS)

#STDP Parameters
taupre = 34 * ms;
taupost = 14 * ms
apre = 103e-2
apost = -40

NumOfDigitsTrain = 800
NumOfDigitsTest = 10

NUM_OUTPUT_CLASSES = len(DIGITS)

timeInh1 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh1.append(3)
    
timeInh2 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh2.append(30)
    
timeInh3 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh3.append(60)
    
timeInh4 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh4.append(90)
    
timeInh5 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh5.append(120)
    
timeInh6 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh6.append(145)

timeInh7 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh7.append(170)

timeExh1 = [3]
timeExh2 = [30]
timeExh3 = [60]
timeExh4 = [90]
timeExh5 = [120]
timeExh6 = [145]
timeExh7 = [170]

Syn12Condition = 'I += 25 * volt/second'
