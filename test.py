from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute
from qiskit.extensions.simulator import snapshot
from qiskit.tools.visualization import circuit_drawer
import numpy as np
import math as m
import scipy as sci
import random
import time
import matplotlib
import matplotlib.pyplot as plt
S_simulator = Aer.backends(name='aer_simulator_statevector')[0]
M_simulator = Aer.backends(name='aer_simulator')[0]



def Measurement(quantumcircuit, **kwargs):
    '''
    Executes a measurement(s) of a QuantumCircuit object for tidier printing
    Keyword Arguments:
    shots (integer) - number of trials to execute for the measurement(s)
    return_M (Bool) - indictaes whether to return the Dictionary object containng measurementprint_M (Bool) - indictaes whether to print the measurement results
    column (Bool) - prints each state in a vertical column
    '''
    p_M = True
    S=1
    ret = False
    NL = False
    if 'shots' in kwargs:
        S = int(kwargs['shots'])
    if 'return_M' in kwargs:
        ret = kwargs['return_M']
    if 'print_M' in kwargs:
        p_M = kwargs['print_M']
    if 'column' in kwargs:
        NL = kwargs['column']
    M1 = execute(quantumcircuit, M_simulator, shots=S).result().get_counts(quantumcircuit)
    M2 = {}
    k1 = list(M1.keys())
    v1 = list(M1.values())
    for k in np.arange(len(k1)):
        key_list = list(k1[k])
        new_key = ''
        for j in np.arange(len(key_list)):
            new_key = new_key+key_list[len(key_list)-(j+1)]
        M2[new_key] = v1[k]
    if(p_M):
        k2 = list(M2.keys())
        v2 = list(M2.values())
        measurements = ''
        for i in np.arange( len(k2) ):
            m_str = str(v2[i])+'|'
            for j in np.arange(len(k2[i])):
                if( k2[i][j] == '0' ):
                    m_str = m_str+'0'
                if( k2[i][j] == '1' ):
                    m_str = m_str+'1'
                if( k2[i][j] == ' ' ):
                    m_str = m_str+'>|'
            m_str = m_str+'>'
            if(NL):
                m_str = m_str + '\n'
            measurements = measurements + m_str
        print(measurements)
    if(ret):
        return M2


def Single_Qubit_Ansatz( qc, qubit, params ):
    '''
    Input:
    qc (QuantumCircuit)
    qubit (QuantumRegister[i])
    params (array)
    Applies the neccessary rotation gates for a single qubit ansatz state
    '''
    qc.ry( params[0], qubit )
    qc.rz( params[1], qubit )

def Two_Qubit_Ansatz(qc, q, params):
    '''
    Input:
    qc (QuantumCircuit)
    q (QuantumRegister)
    params (array)
    Applies the neccessary rotation and CNOT gates for a two qubit ansatz state
    '''
    Single_Qubit_Ansatz( qc, q[0], [params[0], params[1]] )
    Single_Qubit_Ansatz( qc, q[1], [params[2], params[3]] )
    qc.cx( q[0], q[1] )
    Single_Qubit_Ansatz( qc, q[0], [params[4], params[5]] )
    Single_Qubit_Ansatz( qc, q[1], [params[6], params[7]] )

def Calculate_MinMax(V, C_type):
    '''
    Input:
    V (vert)
    C_type (string)
    Returns the smallest or biggest value / index for the smallest value in a list
    '''
    if( C_type == 'min' ):
        lowest = [V[0],0]
        for i in np.arange(1,len(V)):
            if( V[i] < lowest[0] ):
                lowest[0] = V[i]
                lowest[1] = int(i)
        return lowest
    if( C_type == 'max' ):
        highest = [V[0],0]
        for i in np.arange(1,len(V)):
            if( V[i] > highest[0] ):
                highest[0] = V[i]
                highest[1] = int(i)
        return highest

def Compute_Centroid(V):
    '''
    Input:
    V (array)
    Computes and returns the centroid from a given list of values
    '''
    points = len( V )
    dim = len( V[0] )
    Cent = []
    for d in np.arange( dim ):
        avg = 0
        for a in np.arange( points ):
            avg = avg + V[a][d]/points
        Cent.append( avg )
    return Cent

def Reflection_Point(P1, P2, alpha):
    '''
    Input:
    P1 (array)
    P2 (array)
    alpha (float)
    Computes a reflection point from P1 around point P2 by an amount alpha
    '''
    P = []
    for p in np.arange( len(P1) ):
        D = P2[p] - P1[p]
        P.append( P1[p]+alpha*D )
    return P

def VQE_EV(params, Ansatz, H, EV_type, **kwargs):
    '''
    Input:
    params (array)
    Ansatz( Single or Two Qubit Ansatz function)
    H (Dictionary)
    EV_type (string)
    Keyword Arguments:
    shots (integer) - Dictates the number of measurements to use per computation
    Computes and returns the expectation value for a given Hamiltonian and set of theta / phi values
    '''
    Shots = 10000
    if 'shots' in kwargs:
        Shots = int( kwargs['shots'] )
    Hk = list( H.keys() )
    H_EV = 0
    for k in np.arange( len(Hk) ):
        L = list( Hk[k] )
        q = QuantumRegister(len(L))
        c = ClassicalRegister(len(L))
        qc = QuantumCircuit(q,c)
        Ansatz( qc, q, params )
        qc.save_statevector()
        sv0 = execute( qc, S_simulator,shots=1).result().get_statevector(qc)
        if( EV_type == 'wavefunction' ):
            for l in np.arange( len(L)):
                if( L[l] == 'X' ):
                    qc.x( q[int(l)] )
                if( L[l] == 'Y' ):
                    qc.y( q[int(l)] )
                if( L[l] == 'Z' ):
                    qc.z( q[int(l)] )
            qc.save_statevector()
            sv = execute( qc, S_simulator, shots=1 ).result().get_statevector()
            H_ev = 0
            for l2 in np.arange(len(sv)):
                H_ev = H_ev + (np.conj(sv[l2])*sv0[l2]).real
            H_EV = H_EV + H[Hk[k]] * H_ev
        elif(EV_type == 'measure'):
            for l in np.arange( len(L) ):
                if( L[l] == 'X' ):
                    qc.ry(-m.pi/2,q[int(l)])
                if( L[l] == 'Y' ):
                    qc.rx( m.pi/2,q[int(l)])
            qc.measure( q,c )
            M = Measurement( qc, shots=Shots, print_M=False, return_M=True )
            Mk = list( M.keys() )
            H_ev = 0
            for m1 in np.arange(len(Mk)):
                MS = list( Mk[m1] )
                e = 1
                for m2 in np.arange(len(MS)):
                    if( MS[m2] == '1' ):
                        e = e*(-1)
                H_ev = H_ev + e * M[Mk[m1]]
            H_EV = H_EV + H[Hk[k]]*H_ev/Shots
    return H_EV
    
def Nelder_Mead(H, Ansatz, Vert, Val, EV_type):
    '''
    Input:
    H (Dictionary)
    Ansatz( Single or Two Qubit Ansatz function)
    Vert (array)
    Computes and appends values for the next step in the Nelder_Mead Optimization Algorithm
    '''
    alpha = 2.0
    gamma = 2.0
    rho   = 0.5
    sigma = 0.5
    add_reflect = False
    add_expand = False
    add_contract = False
    shrink = False
    add_bool = False
#----------------------------------------
    hi = Calculate_MinMax( Val,'max' )
    Vert2 = []
    Val2 = []
    for i in np.arange(len(Val)):
        if( int(i) != hi[1] ):
            Vert2.append( Vert[i] )
            Val2.append( Val[i] )
    Center_P = Compute_Centroid( Vert2 )
    Reflect_P = Reflection_Point(Vert[hi[1]],Center_P,alpha)
    Reflect_V = VQE_EV(Reflect_P,Ansatz,H,EV_type)
#------------------------------------------------- 
# Determine if: Reflect / Expand / Contract / Shrink
    hi2 = Calculate_MinMax( Val2,'max' )
    lo2 = Calculate_MinMax( Val2,'min' )
    if( hi2[0] > Reflect_V >= lo2[0] ):
        add_reflect = True
    elif( Reflect_V < lo2[0] ):
        Expand_P = Reflection_Point(Center_P,Reflect_P,gamma)
        Expand_V = VQE_EV(Expand_P,Ansatz,H,EV_type)
        if( Expand_V < Reflect_V ):
            add_expand = True
        else:
            add_reflect = True
    elif( Reflect_V > hi2[0] ):
        if( Reflect_V < hi[0] ):
            Contract_P = Reflection_Point(Center_P,Reflect_P,rho)
            Contract_V = VQE_EV(Contract_P,Ansatz,H,EV_type)
            if( Contract_V < Reflect_V ):
                add_contract = True
            else:
                shrink = True
        else:
            Contract_P = Reflection_Point(Center_P,Vert[hi[1]],rho)
            Contract_V = VQE_EV(Contract_P,Ansatz,H,EV_type)
            if( Contract_V < Val[hi[1]] ):
                add_contract = True
            else:
                shrink = True
#-------------------------------------------------
 # Apply: Reflect / Expand / Contract / Shrink
    if( add_reflect == True ):
        new_P = Reflect_P
        new_V = Reflect_V
        add_bool = True
    elif( add_expand == True ):
        new_P = Expand_P
        new_V = Expand_V
        add_bool = True
    elif( add_contract == True ):
        new_P = Contract_P
        new_V = Contract_V
        add_bool = True
    if( add_bool ):
        del Vert[hi[1]]
        del Val[hi[1]]
        Vert.append( new_P )
        Val.append( new_V )
    if( shrink ):
        Vert3 = []
        Val3 = []
        lo = Calculate_MinMax( Val,'min' )
        Vert3.append( Vert[lo[1]] )
        Val3.append( Val[lo[1]] )
        for j in np.arange( len(Val) ):
            if( int(j) != lo[1] ):
                Shrink_P = Reflection_Point(Vert[lo[1]],Vert[j],sigma)
                Vert3.append( Shrink_P )
                Val3.append( VQE_EV(Shrink_P,Ansatz,H,EV_type) )
        for j2 in np.arange( len(Val) ):
            del Vert[0]
            del Val[0]
            Vert.append( Vert3[j2] )
            Val.append( Val3[j2] )