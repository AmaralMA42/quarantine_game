# Simplified program for obtaining the numerical integration of the Quarantine Game version of the SIR model
# This script was developed by Marco Antonio Amaral, as a suplemental material for the related research paper at:
# .... arxiv number
# For more information please contact the author at marcoantonio.amaral@gmail.com
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# SIR model with quarentine (C) and carless (D) strategies
# Mean-field, well-mixed populations
# 5 compartments, Sc, Sd, Ic, Ic and R
# Parameters
quac = 1          			# quarentine cost
disec = 10        			# disease cost
betac = 1.0       			# infection rate for cooperators
betad = 10.0                 	 	# infection rate for defectors
betam = 0.1*(betac+betad)/2   	# cross infection rate
gamma = 1.0         			# recovery rate
tau = 1         			# strategy change time scale
k = 0.1           			# irrationality
tmax = 100         			# maximum time
h = 0.01          			# integration step
C0 = 0.5          			# initial fraction of C
D0 = 1-C0         			# initial fraction of D
S0 = 0.99         			# initial susceptibles
I0 = 1-S0         			# initial fraction of I
R0 = 0            			# initial fraction of R
t0 = 0.0          			# initial time


def model(y, t, quac, disec, betac, betad, betam, gamma, tau, k):
    # function that returns dy/dt
    sd = y[0]
    sc = y[1]
    id = y[2]
    ic = y[3]
    r = y[4]

    uc = -quac				             # cooperator's payoff
    ud = -(id+ic)*betad*disec		     # defector's payoff

    f_ucud = 1/(1+np.exp(-(ud-uc)/k))	 # fermi function
    f_uduc = 1/(1+np.exp(-(uc-ud)/k))	 # fermi function

    phisd = sc*sd*(f_ucud-f_uduc)
    phisd += sc*id*f_ucud-sd*ic*f_uduc  # Phi_S, susceptible, defector
    phiid = ic*id*(f_ucud-f_uduc)
    phiid += ic*sd*f_ucud-id*sc*f_uduc  # Phi_I, infected, defector

    sddot = -sd*(betad*id+betam*ic)+tau*phisd
    scdot = -sc*(betam*id+betac*ic)-tau*phisd
    iddot = sd*(betad*id+betam*ic)-gamma*id+tau*phiid
    icdot = sc*(betam*id+betac*ic)-gamma*ic-tau*phiid
    rdot = gamma*(id+ic)

    dfdt = [sddot, scdot, iddot, icdot, rdot]
    return dfdt


# Initialize values
sd0 = S0*D0
sc0 = S0*C0
id0 = I0*D0
ic0 = I0*C0
r0 = 0
datafilesir = open("SIRCD.dat", "w+")
datafilesir2 = open("SIRCD2.dat", "w+")

# Bundle initial conditions for ODE solver
y0 = [sd0, sc0, id0, ic0, r0]

# Make time array for solution
tStop = tmax
tInc = h
t = np.arange(0., tStop, tInc)

# Call the ODE solver
psoln = odeint(model, y0, t, args=(quac, disec, betac, betad, betam, gamma, tau, k))

# Plot subpopulations as a function of time
fig1 = plt.figure(1, figsize=(8, 8))
plt.plot(t, psoln[:, 0], label='Sd')
plt.plot(t, psoln[:, 1], label='Sc')
plt.plot(t, psoln[:, 2], label='id')
plt.plot(t, psoln[:, 3], label='ic')
plt.plot(t, psoln[:, 4], label='R')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Densities')
plt.title('Subpopulation time evolution')

# Plot populations as a function of time
fig2 = plt.figure(2, figsize=(8, 8))
plt.plot(t, psoln[:, 0]+psoln[:, 1], label='S')
plt.plot(t, psoln[:, 2]+psoln[:, 3], label='I')
plt.plot(t, psoln[:, 4], label='R')
plt.xlabel('Time')
plt.ylabel('Densities')
plt.title('Population time evolution')
plt.legend()

fig3 = plt.figure(3, figsize=(8, 8))
# Plot strategies as a function of time
plt.plot(t, (psoln[:, 0]+psoln[:, 2])/(psoln[:, 0]+psoln[:, 2]+psoln[:, 1]+psoln[:, 3]), label='D')
plt.plot(t, (psoln[:, 1]+psoln[:, 3])/(psoln[:, 0]+psoln[:, 2]+psoln[:, 1]+psoln[:, 3]), label='C')
plt.xlabel('Time')
plt.ylabel('Densities')
plt.title('Strategies time evolution')
plt.legend()

plt.tight_layout()
plt.show()
