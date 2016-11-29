#!/bin/python
#-----------------------------------------------------------------------------
# File Name : brian2_lif.py
# Author: Emre Neftci
#
# Creation Date : Wed 28 Sep 2016 12:05:29 PM PDT
# Last Modified : Fri 18 Nov 2016 01:49:09 PM PST
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
#Modified from Brian2 documentation examples 
from brian2 import *
from npamlib import *

Cm = 50*pF; gl = 1e-9*siemens; taus = 20*ms
Vt = 50*mV; Vr = 0*mV; 
#STDP Parameters
taupre = 20*ms; taupost = taupre
apre = 1.0e-12; apost = apre * taupre / taupost * 1.0
alpha = 0.01


eqs = '''
dv/dt  = -gl*v/Cm + isyn/Cm: volt (unless refractory)
disyn/dt  = -isyn/taus : amp 
'''

data, labels = data_load_mnist([0,1,2,3,4,5,6,7,8,9]) #create 200 mnist data samples
data = (data.T/(data.T).sum(axis=0)).T

duration = 250*ms

## Spiking Network
#Following 2 lines for time-dependent inputs
rate = TimedArray(data*5000*Hz, dt = duration)
Pin = NeuronGroup(data.shape[1], 'rates = rate(t,i) : Hz', threshold='rand()<rates*dt')
P = NeuronGroup(81, eqs, threshold='v>Vt', reset='v = Vr',
                refractory=4*ms, method='euler')

Sff = Synapses(Pin, P, '''w : 1
                        dx/dt = -x / taupre  : 1 (event-driven)
                        dy/dt = -y / taupost : 1 (event-driven)''',
             on_pre='''isyn += w*amp
                        x += apre
                        w += y''',
             on_post='''y += apost
                        w += x-alpha*w''')

Sff.connect() #connect all to all
Sff.w = '(rand()-.5)*5e-9'

wreci = -5 * nA
#Inhibitory connections
Sreci = Synapses(P, P, on_pre='isyn += wreci')
Sreci.connect()

#wrece = .02 * nA
#Excitatory connections
#Srece = Synapses(P, P, on_pre='isyn += wrece')
#Srece.connect(condition='abs(i-j)<=10') #connect to nearest 10

s_monin = SpikeMonitor(Pin)
s_mon = SpikeMonitor(P)

#v_mon = StateMonitor(P, variables='v', record = [0])
#isyn_mon = StateMonitor(P, variables='isyn', record = [0])

run(50 * second)

figure(figsize=(9,9))
plot(s_mon.t/ms, s_mon.i, '.k')
plot(s_monin.t/ms, s_monin.i, '.b', alpha=.2)
xlabel('Time (ms)')
ylabel('Neuron index')
ylim([-1,len(P)+1])
tight_layout()

figure()
W = np.array(Sff.w).T.reshape(len(Pin),len(P)).T
stim_show(W-.5)


show()
print("completed")
