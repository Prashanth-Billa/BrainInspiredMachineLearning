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

NumOfDigitsTrain = 900
NumOfDigitsTest = 10

SPIRAL_PROCESSING = False

NUM_OUTPUT_CLASSES = len(DIGITS)

timeInh1 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh1.append(2)
    
timeInh2 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh2.append(20)
    
timeInh3 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh3.append(40)
    
timeInh4 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh4.append(60)
    
timeInh5 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh5.append(80)
    
timeInh6 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh6.append(10)

timeInh7 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh7.append(120)
    
timeInh8 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh8.append(140)
    
timeInh9 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh9.append(160)
    
timeInh10 = []
for i in range(0, NUM_OUTPUT_CLASSES-1):
    timeInh10.append(180)

timeExh1 = [2]
timeExh2 = [20]
timeExh3 = [40]
timeExh4 = [60]
timeExh5 = [80]
timeExh6 = [100]
timeExh7 = [120]
timeExh8 = [140]
timeExh9 = [160]
timeExh10 = [180]

Syn12ConditionTraining = 'I += 50 * volt/second'
Syn12ConditionTesting = 'I += 25 * volt/second'
