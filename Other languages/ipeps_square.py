# -*- coding: utf-8 -*-
# doPEPS_TEBD.py
import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
import copy 
from scipy.sparse.linalg import eigs
from ncon import ncon


def doPEPS_TEBD(A,B,chiM,hloc,taustep, Tbnd = [], numiter = 50, 
                breps = 1, updateon = True, dispon = True, enexact = 0.0):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 29/1/2019
------------------------
Implementation of imaginary time evolution (TEBD) for a square-lattice PEPS \
with 2-site unit cell (A-B). Inputs 'A' and 'B' are the initial PEPS tensors. \
The full PEPS environment is contracted using a variant of the corner \
transfer matrix (CTM), where 'chiM' sets the dimension of the boundary \
tensors. The nearest neighbor input Hamiltonian is specified by 'hloc', while \
'taustep' specifies the time-step of the TEBD algorithm. Initial boundary \
tensors can be input in the cell 'Tbnd' (if continuing a PEPS calculation) or \
this argument can be omitted in order to initialize new boundary tensors. \
Outputs the updated PEPS tensors 'A' and 'B', the average energy density \
'avEn', and the set of boundary tensors 'Tbnd'.

Note: PEPS is a temperamental beast; the algorithm can become unstable under \
certain circumstances. Some things to try if the algorithm is not converging: \
(i) decrease time-step "taustep", (ii) increase the number of boundary steps \
"breps".

Optional arguments:
`Tbnd`: set of boundary tensors (i.e. given from previous PEPS contraction)
`numiter::Int=50`: total number of TEBD time-steps to perform
`breps::Int=1`: number of boundary update steps between each time step
`updateon::Bool=true`: specify wether or not to perform PEPS tensor updates
`dispon::Bool=true`: specify wether or not to display convergence data
`enexact::Float=0.0`: specify exact ground energy (if known)
"""
    chid = A.shape[4]
    chiD = A.shape[0]

    ##### Initialize new boundary tensors if they are either not given
    if len(Tbnd) == 0:
        CA = [0 for x in range(4)]; CB = [0 for x in range(4)]
        TA = [0 for x in range(4)]; TB = [0 for x in range(4)]
        for k in range(4):
            CA[k] = np.random.rand(2,2)
            CB[k] = np.random.rand(2,2)
            TA[k] = np.random.rand(2,chiD,chiD,2)
            TB[k] = np.random.rand(2,chiD,chiD,2)
        
        initsteps = 10
    else:
        CA = Tbnd[0]; CB = Tbnd[1]
        TA = Tbnd[2]; TB = Tbnd[3]
        initsteps = 2
    
    for k in range(initsteps):
        for bdir in range(4):
            CA,CB,TA,TB = doBoundCont(A,B,CA,CB,TA,TB,bdir,chiM)

    ##### Exponentiate local Hamiltonian
    ehloc = expm(-taustep*hloc)
    uF,sF,vhF = LA.svd(ehloc.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4))
    chicut = sum(sF > 1e-12)
    hl = (uF[:,:chicut] @ np.diag(np.sqrt(abs(sF[:chicut])))).reshape(2,2,chicut)
    hr = ((vhF[:,:chicut]).T @ np.diag(np.sqrt(abs(sF[:chicut])))).reshape(2,2,chicut)
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0,-1]])
    
    Aold = copy.deepcopy(A)  
    Bold = copy.deepcopy(B)  
    errmany = [0 for x in range(4)]
    enLink = [0 for x in range(4)]
    for p in range(numiter):
        for linkLoc in range(4):
            #### Contract PEPS from boundary (four directions)
            for bdir in range(4):
                for k in range(breps):
                    CA,CB,TA,TB = doBoundCont(A,B,CA,CB,TA,TB,bdir,chiM)
                
            ##### Evaluate energy and observables, display results
            if linkLoc == 0:
                rhotwo = [0 for x in range(4)]
                for k in range(4):
                    lvec = np.mod(np.array([0,1,2,3])+k,4)
                    pvec = np.append(lvec,4)
                    
                    envtemp = ncon([CB[lvec[0]],TB[lvec[0]],CB[lvec[1]],TB[lvec[1]],TA[lvec[1]],
                                    CA[lvec[2]],TA[lvec[2]],CA[lvec[3]],TA[lvec[3]],TB[lvec[3]],
                                    B.transpose(pvec),B.transpose(pvec),A.transpose(pvec),A.transpose(pvec)],
                                    [[5,1],[1,7,9,2],[2,10],[10,11,12,22],[22,19,20,18],[18,4],[4,15,17,3],[3,13],
                                     [13,14,16,21],[21,6,8,5],[6,7,11,-6,-5],[8,9,12,-8,-7],[14,-2,19,15,-1],
                                     [16,-4,20,17,-3]]);

                    rhotemp = ncon([envtemp],[[-1,1,-3,2,-2,1,-4,2]]).reshape(chid**2,chid**2)
                    rhotwo[k] = rhotemp / np.trace(rhotemp)
                    enLink[k] = np.trace(rhotwo[k] @ hloc)
                
                rhoA = ncon([rhotwo[1].reshape(2,2,2,2)],[[1,-1,1,-2]])
                rhoB = ncon([rhotwo[1].reshape(2,2,2,2)],[[-1,1,-2,1]])
                SpontMagA = np.sqrt(abs(np.trace(sX@rhoA / 2))**2 + 
                                    abs(np.trace(sY@rhoA / 2))**2 + abs(np.trace(sZ@rhoA / 2))**2)
                SpontMagB = np.sqrt(abs(np.trace(sX@rhoB / 2))**2 + 
                                    abs(np.trace(sY@rhoB / 2))**2 + abs(np.trace(sZ@rhoB / 2))**2)
                SpontMag = (SpontMagA + SpontMagB)/2
                XMag = np.trace(rhoA @ sX)

                ABconv = max(min(LA.norm(A-Aold),LA.norm(A+Aold)),min(LA.norm(B-Bold),LA.norm(B+Bold)))
                Aold = copy.deepcopy(A)  
                Bold = copy.deepcopy(B)  
                avEn = sum(enLink[:4])/2

                if dispon:
                    # Displays iterations, convergence of PEPS tensors, Energy density, Magnetization
                    print('Iteration: %d of %d, Conv: %2.5f, Energy: %f, Energy Error: %e, S-Mag: %e, X-Mag: %e'
                      % (p,numiter,ABconv,avEn,avEn-enexact,SpontMag,XMag))
                    
            ##### Update a link of the PEPS with TEBD # Full Update - FFU
            if updateon:
                # generate environment of a link
                lvec = np.mod(np.array([0,1,2,3])+linkLoc,4)
                pvec = np.append(lvec,4)
                
                envtemp = ncon([CB[lvec[0]],TB[lvec[0]],CB[lvec[1]],TB[lvec[1]],TA[lvec[1]],
                                CA[lvec[2]],TA[lvec[2]],CA[lvec[3]],TA[lvec[3]],TB[lvec[3]],
                                np.transpose(B,pvec),np.transpose(B,pvec),np.transpose(A,pvec),np.transpose(A,pvec)],
                                [[5,1],[1,7,9,2],[2,10],[10,11,12,22],[22,19,20,18],[18,4],[4,15,17,3],[3,13],
                                 [13,14,16,21],[21,6,8,5],[6,7,11,-6,-5],[8,9,12,-8,-7],[14,-2,19,15,-1],[16,-4,20,17,-3]])
                envham = ncon([envtemp,hr,hr,hl,hl],[[1,-1,3,-5,5,-3,7,-7],[1,2,-2],[3,2,-6],[5,6,-4],
                               [7,6,-8]]).reshape(chicut*chiD,chicut*chiD,chicut*chiD,chicut*chiD);

                # find matrices that implement link truncation
                PL, PR, errmany[linkLoc] = doEnvTrun(envham,chiD)
                
                # do link truncation
                ipvec = np.argsort(pvec)
                Bnew = ncon([np.transpose(B,pvec),hl,PL.reshape(chiD,chicut,chiD)],[[-1,-2,-3,2,1],[1,-5,3],[2,3,-4]])
                Anew = ncon([np.transpose(A,pvec),hr,PR.reshape(chiD,chicut,chiD)],[[-1,2,-3,-4,1],[1,-5,3],[2,3,-2]])
                B = np.sign(sum(Bnew.flatten()))*np.transpose(Bnew,ipvec) / LA.norm(Bnew)
                A = np.sign(sum(Anew.flatten()))*np.transpose(Anew,ipvec) / LA.norm(Anew)
                
            
    avEn = sum(enLink[:4])/2
    Tbnd = [0 for x in range(4)]
    Tbnd[0] = CA; Tbnd[1] = CB;
    Tbnd[2] = TA; Tbnd[3] = TB;

    return A,B,avEn,Tbnd


"""
doEnvTrun: function for truncating an interal link in a tensor network to \
dimension 'chiD' using the link environment 'gamenv'
"""
def doEnvTrun(gamenv,chiD, cuttol = 1e-10, iternum = 10):
    
    # normalize environment
    
    chi = gamenv.shape[0]
 
    gamenv = gamenv / ncon([gamenv],[[1,1,2,2]])
    gamenv = 0.5*(gamenv + gamenv.transpose(2,3,0,1))

    Dtemp, Utemp = LA.eigh(gamenv.reshape(chi**2,chi**2))
    chimid = sum(Dtemp > cuttol)
    gamenv = (Utemp[:,range(-1,-chimid-1,-1)] @ np.diag(Dtemp[range(-1,-chimid-1,-1)]) @
        (Utemp[:,range(-1,-chimid-1,-1)]).T).reshape(chi,chi,chi,chi)

    # initialize tensors
    xvec = np.array(np.zeros(int(chi/chiD))); xvec[0] = 1
    ut = ncon([np.eye(chiD,chiD),xvec],[[-1,-3],[-2]]).reshape(chi,chiD)
    vt = ncon([np.eye(chiD,chiD),xvec],[[-1,-3],[-2]]).reshape(chi,chiD)
    
    # alternating optimization
    for k in range(iternum):
        B = ncon([gamenv,ut,ut],[[-1,1,-3,2],[1,-2],[2,-4]]).reshape(chi*chiD,chi*chiD)
        P = ncon([gamenv,ut],[[-1,2,1,1],[2,-2]]).reshape(chi*chiD)
        uF,sF,vhF = LA.svd((LA.pinv(0.5*(B + B.T),rcond = cuttol) @ P
                            ).reshape(chi,chiD),full_matrices=False)
        vt = uF @ vhF
           
        B = ncon([gamenv,vt,vt],[[1,-2,2,-4],[1,-1],[2,-3]]).reshape(chi*chiD,chi*chiD)
        P = ncon([gamenv,vt],[[2,-2,1,1],[2,-1]]).reshape(chi*chiD)
        uF,sF,vhF = LA.svd(((LA.pinv(0.5*(B + B.T),rcond = cuttol) @ P
                             ).reshape(chiD,chi)).T,full_matrices=False)
        ut = uF @ vhF
    
    B = ncon([gamenv,ut,ut],[[-1,1,-3,2],[1,-2],[2,-4]]).reshape(chi*chiD,chi*chiD)
    P = ncon([gamenv,ut],[[-1,2,1,1],[2,-2]]).reshape(chi*chiD)
    uF,sF,vhF = LA.svd((LA.pinv(0.5*(B + B.T),rcond = cuttol) @ P
                            ).reshape(chi,chiD),full_matrices=False)
    PR = uF @ np.diag(np.sqrt(sF)) @ vhF
    PL = ut @ (vhF.T) @ np.diag(np.sqrt(sF)) @ vhF;

    # gauge fixing step
    gamnew = ncon([gamenv,PR,PL,PR,PL],[[1,2,3,4],[1,-1],[2,-2],[3,-3],[4,-4]])

    dtemp, Ltemp = eigs(gamnew.transpose(1,3,0,2).reshape(chiD**2,chiD**2),k=1)
    dtemp, Rtemp = eigs(gamnew.transpose(0,2,1,3).reshape(chiD**2,chiD**2),k=1)
    xtemp = Ltemp.reshape(chiD,chiD)
    ytemp = Rtemp.reshape(chiD,chiD)

    DL, UL = LA.eigh(0.5*np.real(xtemp+xtemp.T))
    DR, UR = LA.eigh(0.5*np.real(ytemp+ytemp.T))

    uF,sF,vhF = LA.svd(np.diag(np.sqrt(abs(DL))) @ (UL.T) @ UR @ 
                       np.diag(np.sqrt(abs(DR))),full_matrices=False)
    xnI = UL @ np.diag(1/np.sqrt(abs(DL))) @ uF @ np.diag(np.sqrt(sF))
    ynI = np.diag(np.sqrt(sF)) @ vhF @ np.diag(1/np.sqrt(abs(DR))) @ (UR.T)
    schg = np.sign(np.diag(xnI))
    xnI = xnI @ np.diag(schg)
    ynI = np.diag(schg) @ ynI
    PL = PL @ xnI
    PR = PR @ (ynI.T)
    
    # check error
    sigtemp = PL @ PR.T
    t1 = ncon([gamenv],[[1,1,2,2]])
    t2 = ncon([gamenv,sigtemp],[[2,3,1,1],[3,2]])
    t3 = ncon([gamenv,sigtemp,sigtemp],[[1,2,3,4],[2,1],[4,3]])
    errtemp = 1 - (t2**2)/(t1*t3)

    return PL, PR, errtemp


"""
doBoundCont: function for contracting the boundary of an iPEPS. Input \
'bdir' sets the direction to contract from, 'chiM' sets the boundary \
index dimension, and 'stol' is the tolerance for small singular values. 
"""
def doBoundCont(A,B,CA,CB,TA,TB,bdir,chiM, stol = 1e-10):

    chiD = A.shape[0]
    pvecL = np.append(np.mod(np.array([0,1,2,3])+bdir,4),4)
    pvecR = np.append(np.mod(np.array([0,1,2,3])+bdir+1,4),4)
    
    # optimise for A-B truncation
    XAL = ncon([TA[pvecL[3]],CA[pvecL[0]],TA[pvecL[0]],A.transpose(pvecL),A.transpose(pvecL)],
        [[-1,3,5,1],[1,2],[2,4,6,-4],[3,4,-5,-2,7],[5,6,-6,-3,7]]).reshape(
                TA[pvecL[3]].shape[0]*chiD**2,TA[pvecL[0]].shape[3]*chiD**2)
    XBR = ncon([TB[pvecL[0]],CB[pvecL[1]],TB[pvecL[1]],B.transpose(pvecR),B.transpose(pvecR)],
        [[-1,3,5,1],[1,2],[2,4,6,-4],[3,4,-5,-2,7],[5,6,-6,-3,7]]).reshape(
                TB[pvecL[0]].shape[0]*chiD**2,TB[pvecL[1]].shape[3]*chiD**2)
    
    uM, sM, vhM = LA.svd(XAL @ XBR,full_matrices=False)
    chiNew = min(sum(sM > stol),chiM)   
    LAB = (LA.pinv(XAL,rcond = stol) @ uM[:,:chiNew] @ np.diag(np.sqrt(sM[:chiNew]))).reshape(
            TA[pvecL[0]].shape[3],chiD,chiD,chiNew)
    RAB = (LA.pinv(XBR.T,rcond = stol) @ (vhM[:chiNew,:]).T @ np.diag(np.sqrt(sM[:chiNew]))).reshape(
            TB[pvecL[0]].shape[0],chiD,chiD,chiNew)
    
    # optimise for B-A truncation
    XBL = ncon([TB[pvecL[3]],CB[pvecL[0]],TB[pvecL[0]],B.transpose(pvecL),B.transpose(pvecL)],
        [[-1,3,5,1],[1,2],[2,4,6,-4],[3,4,-5,-2,7],[5,6,-6,-3,7]]).reshape(
                TB[pvecL[3]].shape[0]*chiD**2,TB[pvecL[0]].shape[3]*chiD**2)
    XAR = ncon([TA[pvecL[0]],CA[pvecL[1]],TA[pvecL[1]],A.transpose(pvecR),A.transpose(pvecR)],
        [[-1,3,5,1],[1,2],[2,4,6,-4],[3,4,-5,-2,7],[5,6,-6,-3,7]]).reshape(
                TA[pvecL[0]].shape[0]*chiD**2,TA[pvecL[1]].shape[3]*chiD**2)
    
    uM, sM, vhM = LA.svd(XBL @ XAR,full_matrices=False)
    chiNew = min(sum(sM > stol),chiM)   
    LBA = (LA.pinv(XBL,rcond = stol) @ uM[:,:chiNew] @ np.diag(np.sqrt(sM[:chiNew]))).reshape(
            TB[pvecL[0]].shape[3],chiD,chiD,chiNew)
    RBA = (LA.pinv(XAR.T,rcond = stol) @ (vhM[:chiNew,:]).T @ np.diag(np.sqrt(sM[:chiNew]))).reshape(
            TA[pvecL[0]].shape[0],chiD,chiD,chiNew)

    # generate new boundary tensors and normalize
    CB1temp = ncon([CA[pvecL[0]],TA[pvecL[3]],LBA],[[1,2],[-1,3,4,1],[2,3,4,-2]])
    CA1temp = ncon([CB[pvecL[0]],TB[pvecL[3]],LAB],[[1,2],[-1,3,4,1],[2,3,4,-2]])
    CB2temp = ncon([CA[pvecL[1]],TA[pvecL[1]],RAB],[[2,1],[1,3,4,-2],[2,3,4,-1]])
    CA2temp = ncon([CB[pvecL[1]],TB[pvecL[1]],RBA],[[2,1],[1,3,4,-2],[2,3,4,-1]])
    TBtemp = ncon([RBA,TA[pvecL[0]],A.transpose(pvecL),A.transpose(pvecL),LAB],
        [[1,2,4,-1],[1,3,5,7],[2,3,8,-2,6],[4,5,9,-3,6],[7,8,9,-4]])
    TAtemp = ncon([RAB,TB[pvecL[0]],B.transpose(pvecL),B.transpose(pvecL),LBA],
        [[1,2,4,-1],[1,3,5,7],[2,3,8,-2,6],[4,5,9,-3,6],[7,8,9,-4]])

    CB[pvecL[0]] = CB1temp /LA.norm(CB1temp)
    CA[pvecL[0]] = CA1temp /LA.norm(CA1temp)
    CB[pvecL[1]] = CB2temp /LA.norm(CB2temp)
    CA[pvecL[1]] = CA2temp /LA.norm(CA2temp)
    TA[pvecL[0]] = TAtemp /LA.norm(TAtemp)
    TB[pvecL[0]] = TBtemp /LA.norm(TBtemp)

    return CA, CB, TA, TB