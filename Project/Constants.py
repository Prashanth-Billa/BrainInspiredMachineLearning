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

NumOfDigitsTrain = 750
NumOfDigitsTest = 10

NUM_OUTPUT_CLASSES = len(DIGITS)

timeInh1 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh1.append(3)
    
timeInh2 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh2.append(40)
    
timeInh3 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh3.append(70)
    
timeInh4 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh4.append(100)
    
timeInh5 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh5.append(130)
    
timeInh6 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh6.append(160)
    


timeExh1 = [3]
timeExh2 = [40]
timeExh3 = [70]
timeExh4 = [100]
timeExh5 = [130]
timeExh6 = [160]

Syn12Condition = 'I += 50 * volt/second'
