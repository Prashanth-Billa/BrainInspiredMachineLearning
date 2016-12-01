from brian2 import *

Cm = 50*pF; gl = 1e-9*siemens; taus = 20*ms
Vt = 50*mV; Vr = 0*mV; 


CLEquations = '''
dv/dt  = -gl*v/Cm + isyn/Cm: volt (unless refractory)
disyn/dt  = -isyn/taus : amp 
'''

threshold='v>Vt'
reset='v = Vr'
refractory=4*ms
method='euler'