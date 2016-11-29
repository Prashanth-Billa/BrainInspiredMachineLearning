from ProjectUtil import *
from Izhikevich import *
from Constants import *
import numpy as np

# ------------------------layer 1 dynamics-------------------------#

data, labels = data_load_mnist(DIGITS)
data = (data.T / (data.T).sum(axis=0)).T

inarr = []
tarr = []

start = data.shape[0] - 50
for index in range(start, start + NumOfDigitsTest):
    print "Label : {0}".format(labels[index - start])
    ret = np.array(np.nonzero(data[index - start].reshape(28, 28))).T
    indicesArray = np.array(ret[:, 0])
    timeArray = np.array(ret[:, 1]) + ((index - start) * DIGIT_DURATION / ms)
    inarr.extend(indicesArray)
    tarr.extend(timeArray)

P1st = SpikeGeneratorGroup(M, inarr, tarr * ms)

# ------------------------layer 2 dynamics-------------------------#

P2nd = NeuronGroup(N/ K_VALUE, IzhikevichEquations, threshold=threshold, reset=reset)

# --------------------connecting layer 1 and layer 2-------------------#
syn12 = Synapses(P1st, P2nd, on_pre=Syn12Condition)
syn12.connect("i/K_VALUE == j")


# ------------------------layer 3/op dynamics-------------------------#


P3rd = NeuronGroup(NUM_OUTPUT_CLASSES, IzhikevichEquations, threshold=threshold, reset=reset)

syn23 = Synapses(P2nd, P3rd, '''w : 1
                        ''',
               on_pre='''I += 100 * w * volt/second
                        ''')


syn23.connect()

# syn23.w = [0.008, 0.008, 0.041, 0.028, 0.033, 0.070, 0.109, 0.137, 0.158, 0.155, 0.132, 0.105, 0.008, 0.008,
# 0.009, 0.010, 0.062, 0.147, 0.100, 0.034, 0.026, 0.051, 0.147, 0.190, 0.142, 0.063, 0.009, 0.009,
# 0.009, 0.009, 0.130, 0.190, 0.083, 0.045, 0.076, 0.062, 0.018, 0.026, 0.122, 0.186, 0.033, 0.010,
# 0.010, 0.010, 0.010, 0.028, 0.082, 0.138, 0.187, 0.208, 0.159, 0.055, 0.036, 0.038, 0.029, 0.010,
# 0.008, 0.008, 0.020, 0.065, 0.092, 0.117, 0.137, 0.152, 0.051, 0.105, 0.144, 0.088, 0.008, 0.008,
# 0.009, 0.033, 0.051, 0.043, 0.051, 0.071, 0.132, 0.165, 0.171, 0.151, 0.097, 0.011, 0.009, 0.009,
# 0.010, 0.010, 0.010, 0.079, 0.199, 0.114, 0.069, 0.045, 0.051, 0.044, 0.066, 0.091, 0.151, 0.063,
# 0.006, 0.006, 0.032, 0.084, 0.114, 0.127, 0.113, 0.055, 0.089, 0.113, 0.113, 0.112, 0.029, 0.006,
# 0.008, 0.008, 0.008, 0.059, 0.133, 0.154, 0.152, 0.155, 0.052, 0.027, 0.041, 0.072, 0.124, 0.008,
# 0.008, 0.008, 0.021, 0.047, 0.083, 0.092, 0.122, 0.126, 0.114, 0.146, 0.156, 0.061, 0.008, 0.008]

syn23.w = np.genfromtxt("trainedWeights.txt")

v_mon = getStateMonitor(P3rd)['voltage']
isyn_mon = getStateMonitor(P3rd)['current']
s_mon = getSpikeMonitor(P3rd)

run(DIGIT_DURATION*NumOfDigitsTest)

print "Test Error  : {0}".format(getError(s_mon, labels, 1))
figure(figsize=(6,4))
plot(s_mon.t/ms, s_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
ylim([-1,len(P1st)+1])
tight_layout()

figure(figsize=(9, 9))
ax = axes()

ax2 = ax.twinx()
ax2.plot(isyn_mon.t / ms, isyn_mon.I[0], 'b', linewidth=3, alpha=.4)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Synaptic Current')
tight_layout()
show()