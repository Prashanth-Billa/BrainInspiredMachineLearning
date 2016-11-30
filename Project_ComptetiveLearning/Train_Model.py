from ProjectUtil import *
from CompetetiveLearning import *
from Constants import *
import numpy as np

# ------------------------layer 1 dynamics-------------------------#

# data, labels = load_mnist_60000('training', digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], path=os.path.dirname(os.path.abspath(__file__)))  #200 sample
# data = (data.T / (data.T).sum(axis=0)).T

data, labels = data_load_mnist(DIGITS)
data = (data.T / (data.T).sum(axis=0)).T

inarr = []
tarr = []

for index in range(0, NumOfDigitsTrain):
    print "Label : {0}".format(labels[index])
    ret = np.array(np.nonzero(data[index].reshape(28, 28))).T
    indicesArray = np.array(ret[:, 0])
    timeArray = np.array(ret[:, 1]) + (index * DIGIT_DURATION/ms)
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


inhNeurons = []
inhNeuronTime = []
exNeurons = []
exNeuronsTime = []
exarr = []

for index in range(0, NumOfDigitsTrain):
    label = labels[index]
    exarr = []
    exarr.append(DIGITS.index(label))
    inhNeurons.extend(getIndicesInh(label))
    inhNeuronTime.extend([(x + index*DIGIT_DURATION/ms) for x in timeInh])
    exNeurons.extend(exarr)
    exNeuronsTime.extend([(x + index*DIGIT_DURATION/ms) for x in timeExh])


Pinh = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, inhNeurons, inhNeuronTime * ms)
PExh = SpikeGeneratorGroup(NUM_OUTPUT_CLASSES, exNeurons, exNeuronsTime * ms)

sinh = Synapses(Pinh, P3rd, on_pre='isyn -= 500*nA')
sinh.connect('i==j')


sinex = Synapses(PExh, P3rd, on_pre='isyn += 500*nA')
sinex.connect('i==j')

v_mon = getStateMonitor(P3rd)['voltage']
isyn_mon = getStateMonitor(P3rd)['current']
s_mon = getSpikeMonitor(P3rd)
s_mon_3 = getSpikeMonitor(P3rd)

weightmon = StateMonitor(syn23, variables=['w'], record=[22])

run(DIGIT_DURATION*NumOfDigitsTrain)
print "Finished training {0} number ".format(NumOfDigitsTrain)
print "************"
print "Training Error  : {0}".format(getError(s_mon, labels))
print syn23.w
figure(figsize=(6,4))
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