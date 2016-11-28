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
for index in range(start, start + NumOfDigits):
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

for x in range(0, K_VALUE):
    syn12.connect(i=[x * (N/K_VALUE), (x*(N/K_VALUE))+(N/K_VALUE) -1], j=x)


# ------------------------layer 3/op dynamics-------------------------#


P3rd = NeuronGroup(NUM_OUTPUT_CLASSES, IzhikevichEquations, threshold=threshold, reset=reset)

syn23 = Synapses(P2nd, P3rd, '''w : 1
                        ''',
               on_pre='''I += 500 * w * volt/second
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

syn23.w = [  3.74775935,   1.63389842,   1.64802591,   1.63952004,
         1.63103   ,   1.63103   ,   1.63103   ,   1.63103   ,
         1.63103   ,   1.63103   ,   3.71475935,   1.60089842,
         2.64672381,   1.65489015,   1.59816443,   1.59803037,
         2.82312202,   1.68458705,   1.59827169,   1.59803067,
         9.17601179,   3.59479416,   7.469721  ,   6.52388465,
         1.64901477,   4.11000754,   6.49422992,   1.52813688,
         4.4262091 ,   2.15086488,  19.27333666,   4.38970701,
        18.91447712,  24.51086394,   5.88025854,   9.39853024,
         6.06594431,   4.80537882,  13.95576041,   4.76321406,
        23.24785509,   4.97143902,  19.13338233,  18.3474746 ,
         8.88082302,  10.76951623,   4.91429068,  16.46636648,
        11.76810907,  14.92978328,  22.84978856,   5.31672521,
         9.5552592 ,   7.02814557,  12.87127537,   9.53197207,
         6.86897968,  21.04348627,  12.25451893,  14.67110086,
        20.01596287,   5.12997229,   6.60900553,  13.18479523,
        16.98868512,   9.420358  ,  14.43289414,  15.58571874,
        11.76627756,  16.95401706,  16.12484449,   4.32483274,
        11.55852337,  13.8637961 ,  25.68751656,   9.17555917,
        18.68316124,   6.3708803 ,   9.54030482,  15.10441455,
        17.21585619,   5.48458839,  20.65383238,   6.10341921,
        20.57723097,   8.78267191,  17.50579796,   7.00456776,
        11.46568612,   7.37286019,  20.67869679,   7.37775595,
        26.98270854,   4.95700726,   8.00113976,  14.07381433,
        18.62823481,   6.35549736,  12.65978916,   2.65575907,
        29.20064535,   6.83112693,  21.86266501,  20.74242623,
         3.01129783,  18.87603131,  15.88639571,   5.93980783,
        14.18445122,   3.50937094,  25.16757447,   5.24796261,
         7.39690882,  24.77669961,   2.83644822,   9.56777898,
         2.39528473,   6.01571407,  16.9617642 ,   5.01981503,
         5.61915435,   1.49882598,   1.81641381,   6.2918337 ,
         1.76219099,   2.48537424,   1.48927256,   6.80815887,
         4.07450396,   5.79304198,   3.74175951,   1.62789842,
         1.64202591,   1.63352004,   1.62503   ,   1.62503   ,
         1.62503   ,   2.32065159,   1.64622583,   1.62508904]

v_mon = getStateMonitor(P3rd)['voltage']
isyn_mon = getStateMonitor(P3rd)['current']
s_mon = getSpikeMonitor(P3rd)

run(DIGIT_DURATION*NumOfDigits)

print "Test Error  : {0}".format(getError(s_mon, labels))
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
ax2.set_ylabel('Synaptic Current [nA]')
tight_layout()
show()