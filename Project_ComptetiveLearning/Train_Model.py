from ProjectUtil import *
from CompetetiveLearning import *
from Constants import *
import numpy as np

# ------------------------layer 1 dynamics-------------------------#

#data, labels = load_mnist_dataset('training', digits=DIGITS)
# data = (data.T / (data.T).sum(axis=0)).T
#data = data.reshape(data.shape[0], 28 * 28)
#labels = labels.reshape(data.shape[0], )

data, labels = data_load_mnist(DIGITS)
data = (data.T / (data.T).sum(axis=0)).T

inarr = []
tarr = []

for index in range(0, NumOfDigitsTrain):
    print "Label : {0}".format(labels[index])
    ret = np.array(np.nonzero(data[index].reshape(28, 28))).T
    indicesArray = np.array(ret[:, 0])
    timeArray = (np.array(ret[:, 1])*6) + (index * DIGIT_DURATION/ms)
    inarr.extend(indicesArray)
    tarr.extend(timeArray)

P1st = SpikeGeneratorGroup(M, inarr, tarr * ms)

# ------------------------layer 2 dynamics-------------------------#

P2nd = NeuronGroup(N/ K_VALUE, CLEquations, threshold=threshold, reset=reset, refractory = refractory,
                   method = method)

# --------------------connecting layer 1 and layer 2-------------------#
syn12 = Synapses(P1st, P2nd, on_pre=Syn12Condition)

# for j in range(0, N/K_VALUE):
#     syn12.connect(i=[x * (N/K_VALUE), (x*(N/K_VALUE))+(N/K_VALUE) -1], j=N/K_VALUE)
syn12.connect("i/K_VALUE == j")

# ------------------------layer 3/op dynamics-------------------------#


P3rd = NeuronGroup(NUM_OUTPUT_CLASSES, CLEquations, threshold=threshold, reset=reset, refractory = refractory,
                   method = method)

syn23 = Synapses(P2nd, P3rd, '''w : 1
                        dx/dt = -x / taupre  : 1 (event-driven)
                        dy/dt = -y / taupost : 1 (event-driven)''',
             on_pre='''isyn += w*amp
                        x += apre
                        w += y''',
             on_post='''y += apost
                        w += x-alpha*w''')
syn23.connect()

# -------------------------competetive learning------------------------#
syn23.w = '(rand()-.5)*5e-9'

wreci = -5 * nA

Sreci = Synapses(P3rd, P3rd, on_pre='isyn += wreci')
Sreci.connect()

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


sinh1 = Synapses(Pinh1, P3rd, on_pre='isyn-= 500*nA')
sinh1.connect('i==j')
sinex1 = Synapses(PExh1, P3rd, on_pre='isyn+=  500*nA')
sinex1.connect('i==j')

sinh2 = Synapses(Pinh2, P3rd, on_pre='isyn-= 500*nA')
sinh2.connect('i==j')
sinex2 = Synapses(PExh2, P3rd, on_pre='isyn+=  500*nA')
sinex2.connect('i==j')

sinh3 = Synapses(Pinh3, P3rd, on_pre='isyn-= 500*nA')
sinh3.connect('i==j')
sinex3 = Synapses(PExh3, P3rd, on_pre='isyn+=  500*nA')
sinex3.connect('i==j')

sinh4 = Synapses(Pinh4, P3rd, on_pre='isyn-=  500*nA')
sinh4.connect('i==j')
sinex4 = Synapses(PExh4, P3rd, on_pre='isyn+=  500*nA')
sinex4.connect('i==j')

sinh5 = Synapses(Pinh5, P3rd, on_pre='isyn-=  500*nA')
sinh5.connect('i==j')
sinex5 = Synapses(PExh5, P3rd, on_pre='isyn+=  500*nA')
sinex5.connect('i==j')

sinh6 = Synapses(Pinh6, P3rd, on_pre='isyn-=  500*nA')
sinh6.connect('i==j')
sinex6 = Synapses(PExh6, P3rd, on_pre='isyn+=  500*nA')
sinex6.connect('i==j')

sinh7 = Synapses(Pinh7, P3rd, on_pre='isyn-=  500*nA')
sinh7.connect('i==j')
sinex7 = Synapses(PExh7, P3rd, on_pre='isyn+=  500*nA')
sinex7.connect('i==j')

v_mon = getStateMonitor(P3rd)['voltage']
isyn_mon = getStateMonitor(P3rd)['current']
s_mon = getSpikeMonitor(P3rd)
s_mon_3 = getSpikeMonitor(P3rd)

weightmon = StateMonitor(syn23, variables=['w'], record=[22])

run(DIGIT_DURATION*NumOfDigitsTrain)
print "Finished training {0} number ".format(NumOfDigitsTrain)
print "************"
print "Training Error  : {0}".format(getError(s_mon, labels))
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
ax2.plot(isyn_mon.t / ms, isyn_mon.isyn[0] / nA, 'b', linewidth=3, alpha=.4)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Synaptic Current [nA]')
tight_layout()
show()