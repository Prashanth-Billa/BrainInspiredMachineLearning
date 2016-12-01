from ProjectUtil import *
from InF import *
from Constants import *
import numpy as np

# ------------------------layer 1 dynamics-------------------------#

data, labels = data_load_mnist(DIGITS)
data = (data.T / (data.T).sum(axis=0)).T

inarr = []
tarr = []

start = 0#data.shape[0] - 50
for index in range(start, start + NumOfDigitsTest):
    print "Label : {0}".format(labels[index - start])
    ret = np.array(np.nonzero(data[index - start].reshape(28, 28))).T
    indicesArray = np.array(ret[:, 0])
    timeArray = np.array(ret[:, 1]) + ((index - start) * DIGIT_DURATION / ms)
    inarr.extend(indicesArray)
    tarr.extend(timeArray)

P1st = SpikeGeneratorGroup(M, inarr, tarr * ms)

# ------------------------layer 2 dynamics-------------------------#

P2nd = NeuronGroup(N/ K_VALUE, InFEquations, threshold=threshold, reset=reset, refractory = refractory,
                   method = method)

# --------------------connecting layer 1 and layer 2-------------------#
syn12 = Synapses(P1st, P2nd, on_pre=Syn12Condition)
syn12.connect("i/K_VALUE == j")


# ------------------------layer 3/op dynamics-------------------------#


P3rd = NeuronGroup(NUM_OUTPUT_CLASSES, InFEquations, threshold=threshold, reset=reset, refractory = refractory,
                   method = method)

#turning off stdp - as it is test phase
apre = 0
apost=0

syn23 = Synapses(P2nd, P3rd, '''w : 1
                        dx/dt = -x / taupre  : 1 (event-driven)
                        dy/dt = -y / taupost : 1 (event-driven)''',
             on_pre='''isyn += w*amp
                        x += apre
                        w += y''')


syn23.connect()

syn23.w = np.genfromtxt("weightsFile.txt")

v_mon = getStateMonitor(P3rd)['voltage']
isyn_mon = getStateMonitor(P3rd)['current']
s_mon = getSpikeMonitor(P3rd)

run(DIGIT_DURATION*NumOfDigitsTest)

print "Accuracy: {0}".format(1 - getError(s_mon, labels, 1))

#debugging
print syn23.w
figure(figsize=(6,4))
plot(s_mon.t/ms, s_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
ylim([-1,len(P1st)+1])
tight_layout()

figure(figsize=(9, 9))
ax = axes()

ax2 = ax.twinx()
ax2.plot(isyn_mon.t / ms, isyn_mon.isyn[0], 'b', linewidth=3, alpha=.4)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Synaptic Current')
tight_layout()
show()
