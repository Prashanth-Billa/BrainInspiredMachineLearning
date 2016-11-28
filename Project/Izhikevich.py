from brian2 import *

a = 0.03e-3/second
b = -2e-3/second
c = -70e-3 * volt
d = 100e-3 * volt/second
CC = 100
I = 0 * volt/second
taus = 10e-3 * second
Vt = -40e-3 * volt
Vr = -60e-3 * volt

IzhikevichEquations = '''
    dv/dt = ((((0.7/ms/mV)*(v-Vr)*(v-Vt)))/CC) - u/CC + I/CC + Isyn/CC          : volt
    du/dt = (a*(b*v-u))  - (a*b*Vr)                                             : volt/second
    dI/dt = -I/taus                                                             : volt/second
    Isyn                                                                        : volt/second
    '''
reset = '''
    v = c
    u += d
    '''
threshold = 'v > Vt'
