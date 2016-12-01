from ProjectUtil import *
from Izhikevich import *
from Constants import *
import numpy as np
# ------------------------layer 1 dynamics-------------------------#

#data, labels = load_mnist_dataset('training', digits=DIGITS)
#data = (data.T / (data.T).sum(axis=0)).T
#data = data.reshape(data.shape[0], 28 * 28)
#labels = labels.reshape(data.shape[0], )

data, labels = data_load_mnist(DIGITS)
data = (data.T / (data.T).sum(axis=0)).T
inarr = []
tarr = []
ret = None

for index in range(0, NumOfDigitsTrain):
    if SPIRAL_PROCESSING == True:
        arr = spiralData(data[index].reshape(28, 28))
        ret = np.array(np.nonzero(np.array(arr).reshape(14, 4*M - 4))).T
    else:
         ret = np.array(np.nonzero(data[index].reshape(28, 28))).T
    indicesArray = np.array(ret[:, 0])
    timeArray = (np.array(ret[:, 1])*6) + (index * DIGIT_DURATION/ms)
    inarr.extend(indicesArray)
    tarr.extend(timeArray)

P1st = SpikeGeneratorGroup(M, inarr, tarr * ms)

# ------------------------layer 2 dynamics-------------------------#

P2nd = NeuronGroup(N/ K_VALUE, IzhikevichEquations, threshold=threshold, reset=reset)

# --------------------connecting layer 1 and layer 2-------------------#
syn12 = Synapses(P1st, P2nd, on_pre=Syn12ConditionTraining)

syn12.connect("i/K_VALUE == j")

# ------------------------layer 3/op dynamics-------------------------#


P3rd = NeuronGroup(NUM_OUTPUT_CLASSES, IzhikevichEquations, threshold=threshold, reset=reset)

syn23 = Synapses(P2nd, P3rd, '''w : 1
                        dx/dt = -x / taupre  : 1 (event-driven)
                             ''',
               on_pre='''x += apre/5
                         I += w * volt/second
                         w += -.00055
                         ''',
               on_post='''w += x - 0.003 +.00055
                        ''')
syn23.connect()


# ---------------inhibitory neuron---------------#

#Excitatory and inhibitory signals at every 20ms of every digit presentation

inhNeurons1 = []
inhNeuronTime1 = []
exNeurons1 = []
exNeuronsTime1 = []
exarr1 = []

inhNeurons2 = []
inhNeuronTime2 = []
exNeurons2 = []
exNeuronsTime2 = []
exarr3 = []

inhNeurons3 = []
inhNeuronTime3 = []
exNeurons3 = []
exNeuronsTime3 = []
exarr3 = []

inhNeurons4 = []
inhNeuronTime4 = []
exNeurons4 = []
exNeuronsTime4 = []
exarr4 = []

inhNeurons5 = []
inhNeuronTime5 = []
exNeurons5 = []
exNeuronsTime5 = []
exarr5 = []

inhNeurons6 = []
inhNeuronTime6 = []
exNeurons6 = []
exNeuronsTime6 = []
exarr6 = []

inhNeurons7 = []
inhNeuronTime7 = []
exNeurons7 = []
exNeuronsTime7 = []
exarr7 = []

inhNeurons8 = []
inhNeuronTime8 = []
exNeurons8 = []
exNeuronsTime8 = []
exarr8 = []

inhNeurons9 = []
inhNeuronTime9 = []
exNeurons9 = []
exNeuronsTime9 = []
exarr9 = []

inhNeurons10 = []
inhNeuronTime10 = []
exNeurons10 = []
exNeuronsTime10 = []
exarr10 = []

for index in range(0, NumOfDigitsTrain):
    label = labels[index]
    exarr1 = []
    exarr1.append(DIGITS.index(label))
    inhNeurons1.extend(getIndicesInh(label))
    inhNeuronTime1.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh1])
    exNeurons1.extend(exarr1)
    exNeuronsTime1.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh1])
    
    exarr2 = []
    exarr2.append(DIGITS.index(label))
    inhNeurons2.extend(getIndicesInh(label))
    inhNeuronTime2.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh2])
    exNeurons2.extend(exarr2)
    exNeuronsTime2.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh2])
    
    exarr3 = []
    exarr3.append(DIGITS.index(label))
    inhNeurons3.extend(getIndicesInh(label))
    inhNeuronTime3.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh3])
    exNeurons3.extend(exarr3)
    exNeuronsTime3.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh3])
    
    exarr4 = []
    exarr4.append(DIGITS.index(label))
    inhNeurons4.extend(getIndicesInh(label))
    inhNeuronTime4.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh4])
    exNeurons4.extend(exarr4)
    exNeuronsTime4.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh4])
    
    exarr5 = []
    exarr5.append(DIGITS.index(label))
    inhNeurons5.extend(getIndicesInh(label))
    inhNeuronTime5.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh5])
    exNeurons5.extend(exarr5)
    exNeuronsTime5.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh5])
    
    exarr6 = []
    exarr6.append(DIGITS.index(label))
    inhNeurons6.extend(getIndicesInh(label))
    inhNeuronTime6.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh6])
    exNeurons6.extend(exarr6)
    exNeuronsTime6.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh6])
    
    exarr7 = []
    exarr7.append(DIGITS.index(label))
    inhNeurons7.extend(getIndicesInh(label))
    inhNeuronTime7.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh7])
    exNeurons7.extend(exarr7)
    exNeuronsTime7.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh7])
    
    exarr8 = []
    exarr8.append(DIGITS.index(label))
    inhNeurons8.extend(getIndicesInh(label))
    inhNeuronTime8.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh8])
    exNeurons8.extend(exarr8)
    exNeuronsTime8.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh8])
    
    exarr9 = []
    exarr9.append(DIGITS.index(label))
    inhNeurons9.extend(getIndicesInh(label))
    inhNeuronTime9.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh9])
    exNeurons9.extend(exarr9)
    exNeuronsTime9.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh9])
    
    exarr10 = []
    exarr10.append(DIGITS.index(label))
    inhNeurons10.extend(getIndicesInh(label))
    inhNeuronTime10.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh10])
    exNeurons10.extend(exarr10)
    exNeuronsTime10.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh10])


Pinh1 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons1, inhNeuronTime1 * ms)
PExh1 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons1, exNeuronsTime1 * ms)

Pinh2 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons2, inhNeuronTime2 * ms)
PExh2 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons2, exNeuronsTime2 * ms)

Pinh3 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons3, inhNeuronTime3 * ms)
PExh3 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons3, exNeuronsTime3 * ms)

Pinh4 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons4, inhNeuronTime4 * ms)
PExh4 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons4, exNeuronsTime4 * ms)

Pinh5 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons5, inhNeuronTime5 * ms)
PExh5 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons5, exNeuronsTime5 * ms)

Pinh6 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons6, inhNeuronTime6 * ms)
PExh6 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons6, exNeuronsTime6 * ms)

Pinh7 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons7, inhNeuronTime7 * ms)
PExh7 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons7, exNeuronsTime7 * ms)

Pinh8 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons8, inhNeuronTime8 * ms)
PExh8 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons8, exNeuronsTime8 * ms)

Pinh9 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons9, inhNeuronTime9 * ms)
PExh9 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons9, exNeuronsTime9 * ms)

Pinh10 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons10, inhNeuronTime10 * ms)
PExh10 = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons10, exNeuronsTime10 * ms)


sinh1 = Synapses(Pinh1, P3rd, on_pre='I -= 1200*volt/second')
sinh1.connect('i==j')
sinex1 = Synapses(PExh1, P3rd, on_pre='I += 750*volt/second')
sinex1.connect('i==j')

sinh2 = Synapses(Pinh2, P3rd, on_pre='I -= 1200*volt/second')
sinh2.connect('i==j')
sinex2 = Synapses(PExh2, P3rd, on_pre='I += 750*volt/second')
sinex2.connect('i==j')

sinh3 = Synapses(Pinh3, P3rd, on_pre='I -= 1200*volt/second')
sinh3.connect('i==j')
sinex3 = Synapses(PExh3, P3rd, on_pre='I += 750*volt/second')
sinex3.connect('i==j')

sinh4 = Synapses(Pinh4, P3rd, on_pre='I -= 1200*volt/second')
sinh4.connect('i==j')
sinex4 = Synapses(PExh4, P3rd, on_pre='I += 750*volt/second')
sinex4.connect('i==j')

sinh5 = Synapses(Pinh5, P3rd, on_pre='I -= 1200*volt/second')
sinh5.connect('i==j')
sinex5 = Synapses(PExh5, P3rd, on_pre='I += 750*volt/second')
sinex5.connect('i==j')

sinh6 = Synapses(Pinh6, P3rd, on_pre='I -= 1200*volt/second')
sinh6.connect('i==j')
sinex6 = Synapses(PExh6, P3rd, on_pre='I += 750*volt/second')
sinex6.connect('i==j')

sinh7 = Synapses(Pinh7, P3rd, on_pre='I -= 1200*volt/second')
sinh7.connect('i==j')
sinex7 = Synapses(PExh7, P3rd, on_pre='I += 750*volt/second')
sinex7.connect('i==j')

sinh8 = Synapses(Pinh8, P3rd, on_pre='I -= 1200*volt/second')
sinh8.connect('i==j')
sinex8 = Synapses(PExh8, P3rd, on_pre='I += 750*volt/second')
sinex8.connect('i==j')

sinh9 = Synapses(Pinh9, P3rd, on_pre='I -= 1200*volt/second')
sinh9.connect('i==j')
sinex9 = Synapses(PExh9, P3rd, on_pre='I += 750*volt/second')
sinex9.connect('i==j')

sinh10 = Synapses(Pinh10, P3rd, on_pre='I -= 1200*volt/second')
sinh10.connect('i==j')
sinex10 = Synapses(PExh10, P3rd, on_pre='I += 750*volt/second')
sinex10.connect('i==j')

v_mon = getStateMonitor(P3rd)['voltage']
isyn_mon = getStateMonitor(P3rd)['current']
s_mon = getSpikeMonitor(P3rd)
s_mon_3 = getSpikeMonitor(P3rd)

weightmon = StateMonitor(syn23, variables=['w'], record=[22])

run(DIGIT_DURATION*NumOfDigitsTrain)
print "Finished training {0} number ".format(NumOfDigitsTrain)
print "************"
#print "Training Error  : {0}".format(getError(s_mon, labels))
weightsFile = open('weightsFile.txt', 'w')
for item in syn23.w:
    weightsFile.write("%s " % item)
weightsFile.close()
print syn23.w
figure(figsize=(10,10))
plot(s_mon.t/ms, s_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
ylim([-1,len(P1st)+1])
tight_layout()

show()
plot(weightmon.t/ms, weightmon.w[0], 'k', linewidth=3, alpha=.6)
show()

figure(figsize=(9, 9))
ax = axes()

ax2 = ax.twinx()
ax2.plot(isyn_mon.t / ms, isyn_mon.I[0] / nA, 'b', linewidth=3, alpha=.4)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Synaptic Current [nA]')
tight_layout()
show()