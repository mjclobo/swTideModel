#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:06:42 2022

1-D single layer shallow water model with variable tidal forcing over domain,
including self-attraction and loading (SAL) effects.
Inviscid, linear, non-rotating.

Goal is balancing low-compute with high-accuracy to predict SAL effects,
updating SAL with expensive calculation infrequently.
Will use:
    - Constant values between updates, as in Barton et al., 2022
        - Pro: Least computations (though minimally)
        - Con: Least accurate by far
    - BC time steps and greater with AB3 predictor scheme
        - Pro: Fewer SAL updates
        - Con: Less accurate 
    - BC time steps (ONLY) to apply predictor-corrector scheme
        - Pro: More accurate
        - Con: Constrained to evaluating every BC time step

@author: Matt Lobo
"""
#%% import modules
import numpy as np
import matplotlib.pyplot as plt

#%% define functions
def rk4(funcIn,psiN,c,dt,dx):
    
    k1     = np.zeros((np.size(psiN)))
    k2     = np.zeros((np.size(psiN)))
    k3     = np.zeros((np.size(psiN)))
    psiNP1 = np.zeros((np.size(psiN)))
    
    for i in np.arange(np.size(psiN)):
        k1[i] = psiN[i] + 0.5*dt*funcIn(psiN,i,c,dx)

    for i in np.arange(np.size(psiN)):
        k2[i] = psiN[i] + 0.5*dt*funcIn(k1,i,c,dx)
    
    for i in np.arange(np.size(psiN)):
        k3[i] = psiN[i] + dt*funcIn(k2,i,c,dx)
        
    for i in np.arange(np.size(psiN)):
        psiNP1[i] = psiN[i] + (1/6)*dt*(funcIn(psiN,i,c,dx) + 2*funcIn(k1,i,c,dx) 
                                        + 2*funcIn(k2,i,c,dx) + funcIn(k3,i,c,dx))
    
    return psiNP1

def cd4H(hN,j0,uN,dx):
    """
    fourth-order centered-difference in space scheme, for our thickness, h
    
    hN: complete spatial vector h for time N
    j0: center index, j, around which you calculate centered difference
    uN: u at time N
    dx: yeah
    
    NOTE: periodic domain implies that indices wrap
    """
    N = np.size(hN)
    
    # use np.mod() to account for indices that go past N-1 (i.e., cyclic boundary conditions)
    m1 = np.mod(j0-1,N-1)
    m2 = np.mod(j0-2,N-1)
    p1 = np.mod(j0+1,N-1)
    p2 = np.mod(j0+2,N-1)
    
    hTerm1 = (4/3)*((hN[p1] - hN[m1])/(2*dx))        # numerator of fourth-order centered diff
    hTerm2 = (1/3)*((hN[p2] - hN[m2])/(4*dx))        # denominator
    
    dhdt = -uN[j0]*(hTerm1-hTerm2)
    
    return dhdt

def cd4U(etaN,j0,uN,dx):
    """
    fourth-order centered-difference in space scheme, for our velocity, u
    
    etaN: complete spatial vector for time N
    j0: center index, j, around which you calculate centered difference
    uN: velocity array at time N
    dx: yeah
    
    NOTE: periodic domain implies that indices wrap
    """
    N = np.size(etaN)
    
    g = 9.81    # m s^-2
    
    # use np.mod() to account for indices that go past N-1 (i.e., cyclic boundary conditions)
    m1 = np.mod(j0-1,N-1)
    m2 = np.mod(j0-2,N-1)
    p1 = np.mod(j0+1,N-1)
    p2 = np.mod(j0+2,N-1)
    
    etaTerm1 = (4/3)*((etaN[p1] - etaN[m1])/(2*dx))        # numerator of fourth-order centered diff
    etaTerm2 = (1/3)*((etaN[p2] - etaN[m2])/(4*dx))        # denominator
    
    dudt = -g*(etaTerm1-etaTerm2)

    return dudt

def execute(params,arraysIn,funcsIn):
    """
    t: time array
    """
    nt = np.size(arraysIn['t'])         # number of time steps
    
    params['evPt'] = np.round(params['nX']/2).astype(int)       # index where we evaluate statistics; just center of domain
    
    etaSALTruet = np.zeros(nt)      # keeping track of all actual evaluated eta_{SAL} values
    etaSALPredt = np.zeros(nt)      # keeping track of all predicted eta_{SAL} values
    sqErrH      = np.zeros(nt)      # calculate square error for RMSE of thickness
    sqErrSAL    = np.zeros(nt)      # calculate square error for RMSE of eta_{SAL}
    
    trueHN = params['H']+arraysIn['hIC']                  # setting initial condition for h 
    trueUN = arraysIn['uIC']                    # setting initial condition for u
    
    predHN = np.copy(trueHN)        # predicted values equal to true values until we have enough BC steps for our scheme
    predUN = np.copy(trueUN)        # predicted values equal to true values until we have enough BC steps for our scheme
    
    nPredSteps  = np.round(params['evalStep']/params['dt']).astype(int) # number of steps we take in between predictions
    n = 0; etaSALPred = 0 # initializing variables
    
    # defining number of evaluated etaSAL values we require to predict future one
    if params['scheme']=='constant':
        nPred = 1
    elif params['scheme']=='AB3':
        nPred = 4
    
    # step thru time
    for i in np.arange(1,nt):
        
        etaEQ,etaSAL    = forc(arraysIn['t'][i],params['amps'],params['oms'],params['nX'])
        etaSALTruet[i]  = etaSAL[params['evPt']]     # time series of true etaSAL values
        etaTrue         = etaEQ+etaSAL
        
        # define eta predicted
        etaSALPred,etaPred,sqErrSAL[i],etaSALPredt[i],n = predEta(i,params,arraysIn,nPredSteps,nPred,etaSALTruet,etaEQ,n,etaSALPred)
        
        # time step actual solution with eta evaluated at each time
        trueUNP1 = funcsIn['timeFunc'](funcsIn['spaceU'],trueHN,trueUN,params['dt'],params['dx'])
        trueHNP1 = funcsIn['timeFunc'](funcsIn['spaceH'],trueHN,trueUN,params['dt'],params['dx']) + etaTrue
        
        # time step predicted solution with predicted eta values
        predUNP1 = funcsIn['timeFunc'](funcsIn['spaceU'],predHN,predUN,params['dt'],params['dx'])
        predHNP1 = funcsIn['timeFunc'](funcsIn['spaceH'],predHN,predUN,params['dt'],params['dx']) + etaPred
        
        # set values for next loop
        trueHN = trueUNP1
        trueUN = trueHNP1
        
        predHN = predUNP1
        predUN = predHNP1
        
        # define squared error for this time step for H
        sqErrH[i]   = (trueHNP1[params['evPt']] - predHNP1[params['evPt']])**2
        
        # plot animation panels if asked to
        if params['plotBool']==True:
            animPlot(i,params,arraysIn,trueHNP1,predHNP1)
    
    # calculate rmse for H and \eta_{SAL}
    rmseH   = np.sqrt(np.mean(sqErrH))
    rmseSAL = np.sqrt(np.mean(sqErrSAL))
    
    # build dict of arrays to return
    valsOut = {'rmseH': rmseH, 'rmseSAL': rmseSAL,
               'trueHFin': trueHN, 'predHFin': predHN,
               'trueUFin': trueUN, 'predUFin': predUN,
               'etaSALTrue': etaSALTruet, 'etaSALPred': etaSALPredt}
    
    return valsOut

def predEta(i,params,arraysIn,nPredSteps,nPred,etaSALTruet,etaEQ,n,etaSALPred):
    
    if i<nPredSteps*nPred:
        etaSALPred  = etaSALTruet[i]
        etaPred     = etaSALPred+etaEQ
        etaSALPredt = etaSALPred
        sqErrSAL    = 0
        n           = 0
    elif np.mod(i,nPredSteps)==0:
        n=0
        etaSALBack = np.zeros(nPred)
        
        for j in np.arange(nPred):
            etaSALBack[j] = etaSALTruet[i-j*nPredSteps]
        
        etaSALPred  = ctrlPred(params,etaSALBack,nPredSteps,nPred,funcSALPred)
        etaPred     = etaEQ+etaSALPred[n]
        sqErrSAL    = (etaSALPred[n] - etaSALTruet[i])**2
        etaSALPredt = etaSALTruet[i]        # we can use true value here being this is the step where we perform full SAL calculation
    else:
        etaPred     = etaEQ+etaSALPred[n]  
        sqErrSAL    = (etaSALPred[n] - etaSALTruet[i])**2
        etaSALPredt = etaSALPred[n]
        n+=1
    
    return etaSALPred,etaPred,sqErrSAL,etaSALPredt,n

def animPlot(i,params,arraysIn,*args):
    if np.mod(i,params['plotPd'])==0:
        names = []
        for j,arg in enumerate(args):
            globals()['y'+str(j+1)] = arg
            names.append('y'+str(j+1))
        
        j    = np.round(i/params['plotPd']).astype(int)
        ymin = params['H']-5
        ymax = params['H']+5
        
        plt.figure(figsize=(30,10))
        for k in names:
            plt.plot(arraysIn['x'],globals()[k],label=k)
        
        plt.axvline(arraysIn['x'][params['evPt']],linestyle='dashed')
        plt.gca().fill_between(arraysIn['x'],y1,ymin, color='blue', alpha=.1)
        plt.legend(fontsize=22)
        plt.grid()
        plt.gca().set_facecolor((0.67,0.84,0.9))
        plt.xlim((np.min(arraysIn['x']),np.max(arraysIn['x'])))
        plt.ylim((ymin,ymax))

        plt.savefig('./figs/anim/test_'+str(j)+'.png')
        plt.close()
        
    return 

def funcSALPred(params,salBack,i):
    # first-order approximation to time derivative of evaluated (true) SAL terms
    # to be used in AB3 scheme
    
    out = (salBack[i]-salBack[i+1])/params['evalStep']
    
    return out

def forc(t,As,oms,nx):
    
    A1,A2,A3,A4,A5      = As[0],As[1],As[2],As[3],As[4]
    om1,om2,om3,om4,om5 = oms[0],oms[1],oms[2],oms[3],oms[4]
    
    f1 = A1*np.cos(om1*t)
    f2 = A2*np.cos(om2*t)
    f3 = A3*np.cos(om3*t)
    
    fSAL = np.ones(nx)*salTerm(nx,t,A4,om4,A5,om5)
    
    fEQ = np.ones(nx)*(f1+f2+f3)
    
    return fEQ,fSAL

def outputs(params,funcsIn):
    
    arraysIn = {}
        
    arraysIn['x'] = np.arange(0,params['dx']*(params['nX']),params['dx'])
    arraysIn['t'] = np.arange(0,3600*params['nHr'],params['dt'])  
    
    arraysIn['uIC'] = funcsIn['ICu'](arraysIn['x'])
    arraysIn['hIC'] = funcsIn['ICh'](arraysIn['x'])
    # arraysIn['hIC'] = np.linspace(0,1,np.size(x)) # have to make boundaries static rather than reentrant
        
    valsOut  = execute(params,arraysIn,funcsIn)
    
    sal = salTerm(np.size(arraysIn['x']),arraysIn['t'],params['amps'][3],params['oms'][3],params['amps'][4],params['oms'][4])
    
    return arraysIn,valsOut,sal

def ctrlPred(params,salBack,nFut,nPred,func):
    """
    Inputs:
        salBack: Array of previous SAL values accounted for on
            baroclinic (longer) time steps
        nFut: How many barotropic steps into the future we want to predict
        nPred: numer of values we have, i.e., len(salBack)
    """
    # pred = np.zeros(nFut)
    
    if params['scheme']=='constant':# hold SAL terms constant in between evaluations
        pred = np.ones(nFut)*salBack[0]
    # Adams-Bashforth 3rd order
    elif params['scheme']=='AB3':
        psiNP1 = salBack[0] + (23/12)*params['evalStep']*func(params,salBack,0) - (16/12)*params['evalStep']*func(params,salBack,1) + (5/12)*params['evalStep']*func(params,salBack,2)
        pred = np.linspace(salBack[0],psiNP1,nFut+1)[1:]
    else:
        raise Exception("You must define a valid scheme using params['scheme'] pair.")
    
    # return predicted SAL values for now until next update
    return pred

def salTerm(nx,t,aP,omP,aSAL,omSAL):
    """
    Here we evaluate our \eta_{SAL} amplitude at time t.
    The variable phase and amplitude arguments represent slight nonlinearities.
    I don't know how realistic they are. I'd need a global tide model for that.

    Parameters
    ----------
    nx : [float]; size of domain
    t : [float]; single value where we are evaluating SAL term
    aP : [float]; amplitude of phase variability (oscillates around 0)
    omP : [float]; time frequency of phase variability
    aSAL : [float]; amplitude of SAL variability (goes around a constant, set here as 0.1)
    omSAL : [float]; time frequency of SAL variability

    Returns
    -------
    etaSAL : [array]; size of spatial domain and constant \eta_{SAL} value for time, t
    
    """
    if np.size(t)==1:
        A      = aSAL*np.cos(t*omP)                     # amplitude of SAL factor as function of time
        phase  = aP*np.cos(t*omP)                       # phase of SAL factor as function of time
        etaSAL = A*np.ones(nx)*np.cos(t*omSAL-phase)    # uniform (across x) SAL term evaluated at time t
    else:
        A      = aSAL*np.cos(t*omP)                 # amplitude of SAL factor as function of time
        phase  = aP*np.cos(t*omP)                   # phase of SAL factor as function of time
        etaSAL = A*np.cos(t*omSAL-phase) 
    
    return etaSAL

#%% define constants (including major tidal frequencies)
# oms   = 2*np.pi*np.array([1/12.4206012,     # M2 0.2
#         1/12,                       # S2
#         1/12.65834751,              # N2
#         1/23.93447213,              # K1
#         1/6.210300601,              # M4 0.05
#         1/25.81933871,              # O1
#         1/22.30608083])/3600.0      # OO1

# amps = np.array([90809,42248,17386,53011,10000,37694,1624])/90809   # tidal potentials taken from Foreman 1977

# define SAL parameters
aP    = 0 #np.pi/8     # amplitude of phase variability
omP   = 0 #1/1.0       # frequency of phase variability
aSAL  = 0.1            # amplitude of \eta_{SAL} (not variability)
omSAL = 1/2       # frequency of \eta_{SAL} variability

# define surface wave parameters
altOm = 2*np.pi*np.array([1/0.5,1/0.5,1/3.0,omP,omSAL])/3600.0  # tidal forcing
altA = np.array([2,0,0,aP,aSAL])

# update increment
evalStep = 10*30 # number of seconds for BC time step (where we loop and evaluate SAL term)

params = {'dx': 30000, 'nX': 100,
         'cour': 0.2, 'H': 4000, 'nHr': 1,
         'scheme': 'AB3', 'evalStep': evalStep,
         'amps': altA, 'oms': altOm,
         'plotPd': 10, 'plotBool': False}

params['dt'] = (params['cour']*params['dx'])/200        # denominator is gravity wave speed

ICu     = lambda x : np.ones(np.size(x))
ICh     = lambda x : np.sin(2*2*np.pi*x/x[-1])
varEQ   = lambda a,x : a*np.sin(2*np.pi*x/x[-1])    # wavenumber 1 variations in equilibrium term

funcsIn = {'spaceU': cd4U, 'spaceH': cd4H, 'timeFunc': rk4,
           'ICu': ICu, 'ICh': ICh, 'varEQ': varEQ}

#%%
# attempt 1
arraysIn,valsOut,sal = outputs(params,funcsIn)

# add curvature factor, i.e., multiply predicted values by some predetermined sinusoidal array which gives curvature to an initial linear interpolation, can be based on derivatives of three past values

#%%
fig = plt.figure(figsize=(20,4))
plt.plot(arraysIn['t'],valsOut['etaSALPred'],linewidth=3.0,label='predicted')
plt.plot(arraysIn['t'],valsOut['etaSALTrue'],linewidth=3.0,label='true')
plt.yticks(fontsize=26)
plt.xticks(fontsize=26)
plt.xlabel('Time [s]',fontsize=30)
plt.ylabel(r'$\eta_{SAL}$ [m]',fontsize=30)
plt.grid()
plt.legend(fontsize=20)
plt.xlim(arraysIn['t'][0],arraysIn['t'][-1])
ax=plt.gca()
# plt.savefig('./figs/AB3Pred.png')



