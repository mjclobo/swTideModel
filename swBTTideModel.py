#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:06:42 2022

1-D single layer shallow water model with variable tidal forcing over domain
(including self-attraction and loading (SAL) effects),
variable wind forcing over domain, and a Rayleigh drag term.
Linear and non-rotating.

Goal is balancing low-compute with high-accuracy to predict SAL effects,
updating SAL with expensive calculation infrequently.
Will use:
    - Constant values between updates, as in Barton et al., 2022
        - Pro: Least computations (though minimally)
        - Con: Least accurate by far
    - BC time steps and greater with AB3 predictor scheme
        - Pro: Fewer SAL updates, simpler scheme than predictor-corrector
        - Con: Less accurate 
    - BC time steps (ONLY) to apply predictor-corrector scheme
        - Pro: More accurate
        - Con: Constrained to evaluating every BC time step, more computations/memory

You currently have to tune wind forcing based on spatial and temporal

@author: Matt Lobo
"""
#%% import modules
import numpy as np
import matplotlib.pyplot as plt

#%% define functions
def rk4(funcIn,psiN,aux,dt,dx):
    
    k1     = np.zeros((np.size(psiN)))
    k2     = np.zeros((np.size(psiN)))
    k3     = np.zeros((np.size(psiN)))
    psiNP1 = np.zeros((np.size(psiN)))
    
    for i in np.arange(np.size(psiN)):
        k1[i] = psiN[i] + 0.5*dt*funcIn(psiN,i,aux,dx)

    for i in np.arange(np.size(psiN)):
        k2[i] = psiN[i] + 0.5*dt*funcIn(k1,i,aux,dx)
    
    for i in np.arange(np.size(psiN)):
        k3[i] = psiN[i] + dt*funcIn(k2,i,aux,dx)
        
    for i in np.arange(np.size(psiN)):
        psiNP1[i] = psiN[i] + (1/6)*dt*(funcIn(psiN,i,aux,dx) + 2*funcIn(k1,i,aux,dx) 
                                        + 2*funcIn(k2,i,aux,dx) + funcIn(k3,i,aux,dx)) 
        
    
    return psiNP1

def eulerStep(funcIn,psiN,aux,dt,dx):
    psiNP1 = np.zeros((np.size(psiN)))
    
    for i in np.arange(np.size(psiN)):
        psiNP1[i] = psiN[i] + dt*funcIn(psiN,i,aux,dx)
    
    return psiNP1

def rk3(funcIn,psiN,c,dt,dx):
    
    psiStar     = np.zeros((np.size(psiN)))
    psiStarStar = np.zeros((np.size(psiN)))
    psiNP1      = np.zeros((np.size(psiN)))
    
    for i in np.arange(np.size(psiN)):
        psiStar[i]     = psiN[i] + (1/3)*dt*funcIn(psiN,i,c,dx)
    
    for i in np.arange(np.size(psiN)):
        psiStarStar[i] = psiStar[i] + (15/16)*dt*(funcIn(psiStar,i,c,dx) - (5/9)*funcIn(psiN,i,c,dx))
    
    for i in np.arange(np.size(psiN)):
        psiNP1[i]   = psiStarStar[i] + (8/15)*dt*((funcIn(psiStarStar,i,c,dx) - (153/128)*(funcIn(psiStar,i,c,dx)-(5/9)*funcIn(psiN,i,c,dx))))
    
    return psiNP1

def cd4H(hN,j0,uN,dx):
    """
    fourth-order centered-difference in space scheme, for our thickness, h
    
    hN: complete spatial vector h for time N
    j0: center index, j, around which you calculate centered difference
    uN: u at time N
    dx: 
    
    NOTE: periodic domain implies that indices wrap
    """
    N = np.size(hN)
    
    # apply cyclic boundary conditions
    m2,m1,p1,p2 = loopIndices(j0,N)
    
    hTerm1 = (4/3)*((uN[p1]*hN[p1] - uN[m1]*hN[m1])/(2*dx))        # numerator of fourth-order centered diff
    hTerm2 = (1/3)*((uN[p2]*hN[p2] - uN[m2]*hN[m2])/(4*dx))        # denominator
    
    dhdt = -(hTerm1-hTerm2)
    
    return dhdt

def cd4U(uN,j0,etaN,dx):
    """
    fourth-order centered-difference in space scheme, for our velocity, u
    
    uN: velocity array at time N
    j0: center index, j, around which you calculate centered difference
    etaN: complete spatial vector for time N
    dx: 
    
    NOTE: periodic domain implies that indices wrap
    """
    N = np.size(etaN)
    
    g = 9.81    # m s^-2
    
    # apply cyclic boundary conditions
    m2,m1,p1,p2 = loopIndices(j0,N)
    
    etaTerm1 = (4/3)*((etaN[p1] - etaN[m1])/(2*dx))        # numerator of fourth-order centered diff
    etaTerm2 = (1/3)*((etaN[p2] - etaN[m2])/(4*dx))        # denominator
    
    dudt = -g*(etaTerm1-etaTerm2)
    
    # KETerm1 = (4/3)*((uN[p1]**2 - uN[m1]**2)/(2*dx))        # nonlinear
    # KETerm2 = (1/3)*((uN[p2]**2 - uN[m2]**2)/(4*dx))        # nonlinear
    
    # dudt = -g*(etaTerm1-etaTerm2) - (KETerm1-KETerm2)       # includes nonlinear KE term
    
    return dudt

def loopIndices(j0,N):
    m1 = np.mod(j0-1,N-1)
    m2 = np.mod(j0-2,N-1)
    p1 = np.mod(j0+1,N-1)
    p2 = np.mod(j0+2,N-1)
    return m2,m1,p1,p2

def execute(params,arraysIn,funcsIn):
    """
    Main loop for running model

    Parameters
    ----------
    params : dict
        Dict of parameters for model.
        Only feasible way of passing so many.
        May try to replace this in the future.
    arraysIn : dict
        Dict of arrays needed to run model.
        Most defined based on params dict in outputs() function.
    funcsIn : dict
        Functions which are needed to initialize/run model (ICs, space and time schemes).

    Returns
    -------
    valsOut : Dict
        Set of prognostic/diagnostic variable arrays output from model.

    """
    
    nt = np.size(arraysIn['t'])         # number of time steps
    
    params['evPt'] = np.round(params['nX']/2).astype(int)       # index where we evaluate statistics; just center of domain
    
    # preallocating memory
    etaSALTruet = np.zeros(nt)      # keeping track of all actual evaluated eta_{SAL} values
    etaSALPredt = np.zeros(nt)      # keeping track of all predicted eta_{SAL} values
    ut          = np.zeros(nt)      # barotropic velocities in time
    ht          = np.zeros(nt)      # SSH in time
    sqErrH      = np.zeros(nt)      # calculate square error for RMSE of thickness
    sqErrSAL    = np.zeros(nt)      # calculate square error for RMSE of eta_{SAL}
    
    # unpacking parameters dict
    H       = params['H']
    dt      = params['dt']
    dx      = params['dx']
    kappa   = params['kappa']
    rho0    = params['rho0']
    tauAmp  = params['tauAmp']
    tauWave = params['tauWavenumber']
    tauOm   = params['tauOm']
    evPt    = params['evPt']
    salFrac = params['salFrac']
    
    x = arraysIn['x']
    t = arraysIn['t']
    
    # setting ICs
    trueHN = H+arraysIn['hIC']        # setting initial condition for h 
    trueUN = arraysIn['uIC']                    # setting initial condition for u
    
    predHN = np.copy(trueHN)        # predicted values equal to true values until we have enough BC steps for our scheme
    predUN = np.copy(trueUN)        # predicted values equal to true values until we have enough BC steps for our scheme
    
    # miscellany
    nPredSteps  = np.round(evalStep/dt).astype(int) # number of steps we take in between predictions
    n = 0; etaSALPred = 0 # initializing variables
    
    # defining number of evaluated etaSAL values we require to predict future one, based on prediction scheme
    if params['scheme']=='constant':
        nPred = 1
    elif params['scheme']=='AB3':
        nPred = 4
    
    # step thru time
    for i in np.arange(1,nt):
        
        # define true equilibrium and SAL terms
        etaEQ,etaSAL    = forc(x,params['tideForcWavenumber'],t[i],params['amps'],params['oms'],params['nX'],salFrac)
        
        # update time series
        etaSALTruet[i]  = etaSAL[evPt]     # time series of true etaSAL values
        
        # define true tidal forcing terms
        etaTidalTrue    = etaEQ+etaSAL
        
        # define eta predicted
        etaSALPred,etaTidalPred,sqErrSAL[i],etaSALPredt[i],n = predEta(i,params,arraysIn,nPredSteps,nPred,etaSALTruet,etaEQ,n,etaSALPred)
        
        # define wind stress (initial conditions just get dissipated by Rayleigh drag...)
        tau = tauAmp*np.cos(tauWave*2*np.pi*(x/x[-1])-tauOm*t[i])
        
        # time step actual solution with eta evaluated at each time
        trueUNP1 = funcsIn['timeFunc'](funcsIn['spaceU'],trueUN,trueHN-etaTidalTrue,dt,dx) - dt*kappa*trueUN + dt*tau/rho0
        trueHNP1 = funcsIn['timeFunc'](funcsIn['spaceH'],trueHN,trueUN,dt,dx)
        
        # time step predicted solution with predicted eta values
        predUNP1 = funcsIn['timeFunc'](funcsIn['spaceU'],predUN,predHN-etaTidalPred,dt,dx) - dt*kappa*predUN + dt*tau/rho0
        predHNP1 = funcsIn['timeFunc'](funcsIn['spaceH'],predHN,predUN,dt,dx)
        
        # set values for next loop
        trueHN = trueHNP1
        trueUN = trueUNP1
        
        predHN = predHNP1
        predUN = predUNP1
        
        # update time arrays
        ut[i] = trueUN[evPt]
        ht[i] = trueHN[evPt]
        
        # define squared error for this time step for H
        sqErrH[i]   = (trueHNP1[evPt] - predHNP1[evPt])**2
        
        # plot animation panels if asked to
        if params['plotBool']==True:
            # animPlot(i,params,arraysIn,trueUN)
            # animPlot(i,params,arraysIn,trueHNP1-etaTidalTrue,predHNP1)    # this looks right...
            animPlot(i,params,arraysIn,trueHNP1,predHNP1)
    
    # calculate rmse for H and \eta_{SAL}
    rmseH   = np.sqrt(np.mean(sqErrH))
    rmseSAL = np.sqrt(np.mean(sqErrSAL))
    
    # build dict of arrays to return
    valsOut = {'rmseH': rmseH, 'rmseSAL': rmseSAL,
               'trueHFin': trueHN, 'predHFin': predHN,
               'trueUFin': trueUN, 'predUFin': predUN,
               'etaSALTrue': etaSALTruet, 'etaSALPred': etaSALPredt,
               'u': ut, 'h': ht}
    
    return valsOut

def predEta(i,params,arraysIn,nPredSteps,nPred,etaSALTruet,etaEQ,n,etaSALPred):
    """
    This is where predictor schemes (or lack thereof) is applied, to define SAL array
    at evPt until next full evaluation.

    Parameters
    ----------
    i : float
        Current time index
    params : dict
        Dict of parameters for model.
    arraysIn : dict
        Dict of arrays needed to run model.
        Most defined based on params dict in outputs() function.
    nPredSteps : float
        Number of steps which we will predict into future
        E.g., might be baroclinic time step divided be delta_time
    nPred : float
        Number of previously evaluated SAl values needed to predict future ones.
        E.g., would be 3 for third-order Adams-Basforth scheme, or 0 if just holding evaluated value constant between updates
    etaSALTruet : float
        True SAl term used to calculate error; at time index i and spatial index evPt
    etaEQ : array (over domain at time step i)
        Equilibrium tidal forcing term (analytical) which is summed with predicted SAL for total tidal forcing
    n : float
        Number of time steps since last update with true SAL term
    etaSALPred : array
        Predicted SAL values for spatial index evPt which are used between true SAL updates

    Returns
    -------
    etaSALPred : array
        Predicted SAL values for spatial index evPt which are used between true SAL updates
    etaTidalPred : array
        Predicted total tidal forcing term for time index i and whole space domain
    sqErrSAL : float
        Square error between predicted and actual SAL values for time index i
    etaSALPredt : float
        Predicted SAL forcing term for time index i and space index evPt
    n : float
        Number of time steps since last update with true SAL term
    """
    
    if i<nPredSteps*nPred:
        etaSALPred   = etaSALTruet[i]
        etaTidalPred = etaSALPred+etaEQ
        etaSALPredt  = etaSALPred
        sqErrSAL     = 0
        n            = 0
    
    elif np.mod(i,nPredSteps)==0:
        n=0
        etaSALBack = np.zeros(nPred)
        
        for j in np.arange(nPred):
            etaSALBack[j] = etaSALTruet[i-j*nPredSteps]
        
        etaSALPred  = ctrlPred(params,etaSALBack,nPredSteps,nPred,funcSALPred)
        etaTidalPred     = etaEQ+etaSALPred[n]
        sqErrSAL    = (etaSALPred[n] - etaSALTruet[i])**2
        etaSALPredt = etaSALTruet[i]        # we can use true value here being this is the step where we perform full SAL calculation
        
    else:
        etaTidalPred     = etaEQ+etaSALPred[n]  
        sqErrSAL    = (etaSALPred[n] - etaSALTruet[i])**2
        etaSALPredt = etaSALPred[n]
        n+=1
    
    return etaSALPred,etaTidalPred,sqErrSAL,etaSALPredt,n

def animPlot(i,params,arraysIn,*args):
    """
    Basic plotting function to create animation frames.
    Currently saves to a hard-coded directory, which can easily be changed to a parameter.

    Parameters
    ----------
    i : float
        time index
    params : dict
        DESCRIPTION.
    arraysIn : dict
        DESCRIPTION.
    *args : args
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if np.mod(i,params['plotPd'])==0:
        names = []; labelNames = []
        
        for j,arg in enumerate(args):
            globals()['y'+str(j+1)] = arg
            names.append('y'+str(j+1))
            if j==0:
                labelNames.append('true SSH')
            else:
                labelNames.append('predicted SSH')
        
        j    = np.round(i/params['plotPd']).astype(int)
        ymin = params['H']-10
        ymax = params['H']+10
        # ymin = -10; ymax = 10
        xIn  = arraysIn['x']/1000.0     # x in km
        plt.figure(figsize=(30,10))
        
        for l,k in enumerate(names):
            plt.plot(xIn,globals()[k],label=labelNames[l])
        
        plt.axvline(xIn[params['evPt']],linestyle='dashed')
        plt.gca().fill_between(xIn,y1,ymin, color='blue', alpha=.1)
        plt.legend(labelNames,fontsize=24)
        plt.grid()
        plt.gca().set_facecolor((0.67,0.84,0.9))
        plt.xlim((np.min(xIn),np.max(xIn)))
        plt.ylim((ymin,ymax))
        plt.yticks(fontsize=26)
        plt.xticks(fontsize=26)
        plt.xlabel('distance [km]',fontsize=30)
        plt.ylabel('SSH [m]',fontsize=30)
        plt.annotate('t='+str((arraysIn['t'][i]/3600).round(decimals=2))+'hr',(0.9,0.75),fontsize=24,xycoords='axes fraction')
        plt.margins(y=0); plt.margins(y=0)  
        
        plt.savefig('./figs/anim/test_'+str(j)+'.png',bbox_inches='tight')
        plt.close()
        
    return

def funcSALPred(params,salBack,i):
    # first-order approximation to time derivative of evaluated (true) SAL terms
    # to be used in AB3 scheme
    
    out = (salBack[i]-salBack[i+1])/params['evalStep']
    
    return out

def forc(x,wav,t,As,oms,nx,salFrac):
    
    A1,A2,A3    = As[0],As[1],As[2]
    om1,om2,om3 = oms[0],oms[1],oms[2]
    
    f1 = A1*np.cos(wav*2*np.pi*(x/x[-1])-om1*t)
    f2 = A2*np.cos(om2*t)
    f3 = A3*np.cos(om3*t)
    
    # SAl forcing, scalar multiple of tidal forcing
    fSAL = salTerm(x,t,salFrac*A1,om1,salFrac*A2,om2,wav)
    
    fEQ = (f1+f2+f3)
    
    return fEQ,fSAL

def outputs(params,funcsIn):
    
    arraysIn = {}
        
    arraysIn['x'] = np.arange(0,params['dx']*(params['nX']),params['dx'])
    arraysIn['t'] = np.arange(0,3600*params['nHr'],params['dt'])  
    
    arraysIn['uIC'] = funcsIn['ICu'](arraysIn['x'])
    arraysIn['hIC'] = funcsIn['ICh'](arraysIn['x'])
    # arraysIn['hIC'] = np.linspace(0,1,np.size(x)) # have to make boundaries static rather than reentrant
        
    valsOut  = execute(params,arraysIn,funcsIn)
        
    return arraysIn,valsOut

def ctrlPred(params,salBack,nFut,nPred,func):
    """
    Inputs:
        salBack: Array of previous SAL values accounted for on
            baroclinic (longer) time steps
        nFut: How many barotropic steps into the future we want to predict
        nPred: numer of values we have, i.e., len(salBack)
    """
    
    # hold SAL terms constant in between evaluations
    if params['scheme']=='constant':
        pred = np.ones(nFut)*salBack[0]
    
    # Adams-Bashforth 3rd order
    elif params['scheme']=='AB3':
        psiNP1 = salBack[0] + (23/12)*params['evalStep']*func(params,salBack,0) - (16/12)*params['evalStep']*func(params,salBack,1) + (5/12)*params['evalStep']*func(params,salBack,2)
        pred = np.linspace(salBack[0],psiNP1,nFut+1)[1:]
    
    else:
        raise Exception("You must define a valid scheme using params['scheme'] dict pair.")
    
    # return predicted SAL values for now until next update
    return pred

def salTerm(x,t,a1,om1,a2,om2,wav):
    
    etaSAL = a1*np.cos(wav*2*np.pi*(x/x[-1])-om1*t) + a2*np.cos(wav*2*np.pi*(x/x[-1])-om2*t)
    
    return etaSAL

#%% define constants and other parameters
# define SAL parameters
salFrac = 0.1 

# define tidal equilibrium parameters
altOm = 2*np.pi*np.array([1/12.42,1/23.93,1/12.0])/3600.0  # tidal forcing: M2, K1, S2
altA = np.array([10,0,0]) # 0.242334

# update increment
evalStep = 1*3600 # number of seconds for BC time step (where we loop and evaluate SAL term)

# parameters dict
params = {'dx': 50000, 'nX': 200,
          'cour': 0.1, 'H': 4000, 'nHr': 2*12,
          'scheme': 'AB3', 'evalStep': evalStep, 'salFrac': salFrac,
          'amps': altA, 'oms': altOm, 'rho0':1025.0,
          'kappa': 1/(0.5*3600), 'tideForcWavenumber': 1,
          'tauWavenumber':4,'tauAmp':0.15,'tauOm':(2*np.pi)/(1.5*3600),
          'plotPd': 10, 'plotBool': False}

params['dt'] = (params['cour']*params['dx'])/200        # denominator is approx. gravity wave speed

# initial conditions for u and h: a stable model requires these being close to their expected values
ICu     = lambda x : np.ones(np.size(x))*np.sqrt(9.81*params['H'])
ICh     = lambda x : np.sin(4*2*np.pi*x/x[-1])*0        # mean depth, H, added elsewhere

# various functions to be used
funcsIn = {'spaceU': cd4U, 'spaceH': cd4H, 'timeFunc': eulerStep,
           'ICu': ICu, 'ICh': ICh}

#%% run model
arraysIn,valsOut = outputs(params,funcsIn)

# idea: add curvature factor, i.e., multiply predicted values by some predetermined sinusoidal array which gives curvature to an initial linear interpolation, can be based on derivatives of three past values

#%% plot results
fig = plt.figure(figsize=(20,4))
plt.plot(arraysIn['t']/3600.0,valsOut['etaSALPred'],linewidth=3.0,label='predicted')
plt.plot(arraysIn['t']/3600.0,valsOut['etaSALTrue'],linewidth=3.0,label='true')
plt.yticks(fontsize=26)
plt.xticks(fontsize=26)
plt.xlabel('Time [hr]',fontsize=30)
plt.ylabel(r'$\eta_{SAL}$ [m]',fontsize=30)
plt.grid()
plt.legend(fontsize=20)
plt.xlim(arraysIn['t'][0]/3600.0,arraysIn['t'][-1]/3600.0)
ax=plt.gca()
plt.savefig('./figs/AB3Pred.png')



