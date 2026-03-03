import equinox as eqx
from jax.lax import cond
import jax.numpy as jnp

class levchenko_2000(eqx.Module):
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=True): # defaults to sustained stim
        self.transient = transient

    def __call__(self, t, y, args):
        # unpack state
        RAFact, RAF, RAF_RAFact, RAFstar, RAFstar_RAFPase, MEK, MEK_RAFstar, MEKstar, \
            MEKstar_MEKPase, MEKstar_RAFstar, MEKstarstar, MEKstarstar_MEKPase, \
            MAPK, MAPK_MEKstarstar, MAPKstar, MAPKstar_MEKstarstar, MAPKstarstar, \
            MAPKstar_MAPKPase, MAPKstarstar_MAPKPase, C2, C3, C4, \
            C5, C6, C7, C8, C9 = y
        
        # unpack parameters
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, d1, d2, d3, d4, d5, d6, d7, d8, \
            d9, d10, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, kOn1, kOn2, kOff1, \
            kOff2, kOff3, kOff4, RAFPase, MEKPase, MAPKPase, total_scaffold = args

        # define additional parameters
        kr1 = k5
        kr2 = k9 

        # algrebraic equations for C1
        C1 = total_scaffold - (C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9)

        # fluxes
        J1 = a1*RAF*RAFact
        J2 = d1*RAF_RAFact
        J3 = k2*RAFstar_RAFPase
        J4 = k1*RAF_RAFact
        J5 = a2*RAFstar*(RAFPase - RAFstar_RAFPase)
        J6 = d2*RAFstar_RAFPase
        J7 = k3*MEK_RAFstar
        J8 = d3*MEK_RAFstar
        J9 = a3*RAFstar*MEK
        J10 = k5*MEKstar_RAFstar
        J11 = d5*MEKstar_RAFstar
        J12 = a5*MEKstar*RAFstar
        J13 = k4*MEKstar_MEKPase
        J14 = kOff1*(C2 + C6 + C9)
        J15 = kOn1*MEK*(C1 + C4 + C5)
        J16 = a4*MEKstar*(MEKPase - MEKstar_MEKPase - MEKstarstar_MEKPase)
        J17 = d4*MEKstar_MEKPase
        J18 = k6*MEKstarstar_MEKPase
        J19 = a6*MEKstarstar*(MEKPase - MEKstar_MEKPase - MEKstarstar_MEKPase)
        J20 = d6*MEKstarstar_MEKPase
        J21 = a7*MEKstarstar*MAPK
        J22 = d7*MAPK_MEKstarstar
        J23 = k7*MAPK_MEKstarstar
        J24 = d9*MAPKstar_MEKstarstar
        J25 = k9*MAPKstar_MEKstarstar
        J26 = a9*MAPKstar*MEKstarstar
        J27 = kOff3*(C3 + C7 + C8)
        J28 = k8*MAPKstar_MAPKPase
        J29 = kOff2*(C4 + C6 + C7)
        J30 = kOn2*MAPK*(C1 + C2 + C3)
        J31 = a8*MAPKstar*(MAPKPase - MAPKstar_MAPKPase - MAPKstarstar_MAPKPase)
        J32 = d8*MAPKstar_MAPKPase
        J33 = k10*MAPKstarstar_MAPKPase
        J34 = a10*MAPKstarstar*(MAPKPase - MAPKstar_MAPKPase - MAPKstarstar_MAPKPase)
        J35 = d10*MAPKstarstar_MAPKPase
        J36 = kOff4*(C5 + C8 + C9)
        J37 = kOn1*C1*MEK
        J38 = kOn2*C1*MAPK
        J39 = kOff1*C2
        J40 = kOff2*C4
        J41 = kOff3*C3
        J42 = kOff4*C5
        J43 = kOff2*C6
        J44 = kOff4*C9
        J45 = kOn2*C2*MAPK
        J46 = kr1*C2*RAFstar
        J47 = kOn2*C3*MAPK
        J48 = kOff2*C7
        J49 = kOff4*C8
        # J50 = kr1*C2*RAF # i think this is wrong
        # skip J50, typo 
        J51 = kOff1*C6
        J52 = kOff3*C7
        J53 = kOn1*C4*MEK
        J54 = kOff1*C9
        J55 = kOff3*C8
        J56 = kOn1*C5*MEK
        # skip J57, typo
        J58 = kr1*C6*RAFstar 
        # J58 = kr1*C6*RAF
        J59 = kr2*C7
        J60 = kr1*C9*RAFstar 
        # J60 =kr1*C9*RAF


        

        
        # ODE rhs
        trans_fun = lambda J1, J2, J4: jnp.squeeze(-J1 + J2 + J4)
        sus_fun = lambda J1, J2, J4: 0.0
        d_RAFact = cond(self.transient, trans_fun, sus_fun, J1, J2, J4)
        d_RAF = -J1 + J2 + J3
        d_RAF_RAFact = J1 - J2 - J4
        d_RAFstar = -J5 + J6 + J4 + J7 + J8 - J9 + J10 + J11 - J12
        d_RAFstar_RAFPase = J5 - J6 - J3
        d_MEK = -J9 + J8 + J13 + J14 - J15
        d_MEK_RAFstar = J9 - J7 - J8
        d_MEKstar = -J16 + J17 + J7 + J18 + J11 - J12
        d_MEKstar_MEKPase = J16 - J17 - J13
        d_MEKstar_RAFstar = J12 - J11 - J10
        d_MEKstarstar = J10 - J19 + J20 - J21 + J22 + J23 + J24 + J25 - J26 + J27
        d_MEKstarstar_MEKPase = J19  - J20 - J18
        d_MAPK = -J21 + J22 + J28 + J29 - J30
        d_MAPK_MEKstarstar = J21 - J22 - J23
        d_MAPKstar = J23 - J31 + J32 - J26 + J24 + J33
        d_MAPKstar_MEKstarstar = J26 - J24 - J25
        d_MAPKstarstar = -J34 + J35 + J25 + J36
        d_MAPKstar_MAPKPase = J31  - J32 - J28
        d_MAPKstarstar_MAPKPase = J34 - J33 - J35
        # d_C1 = -J37 - J38 + J39 + J40 + J41 + J42
        d_C2 = J37 + J43 + J44 - J39 - J45 - J46
        d_C3 = -J47 + J48 - J41 + J49 + J46
        d_C4 = J51 + J38 + J52 - J40 - J53
        d_C5 = J54 + J55 - J56 - J42
        d_C6 = J53 + J45 - J51 - J43 - J58
        d_C7 = -J52 + J47 - J59 + J58 - J48
        d_C8 = J59 - J55 - J49 + J60
        d_C9 = J56 - J54 - J44 - J60
        
        # concatenate into tuple and return
        return (d_RAFact, d_RAF, d_RAF_RAFact, d_RAFstar, d_RAFstar_RAFPase, d_MEK, 
                d_MEK_RAFstar, d_MEKstar, d_MEKstar_MEKPase, d_MEKstar_RAFstar, 
                d_MEKstarstar, d_MEKstarstar_MEKPase, d_MAPK, d_MAPK_MEKstarstar, 
                d_MAPKstar, d_MAPKstar_MEKstarstar, d_MAPKstarstar, 
                d_MAPKstar_MAPKPase, d_MAPKstarstar_MAPKPase, # d_C1, 
                d_C2, d_C3, d_C4, d_C5, d_C6, d_C7, d_C8, d_C9)
    

    def get_nominal_params(self):
        # values come from Ferrell 1996 - trends biochem sci
        #   and Bray and Lay 1997 - PNAS 
        p_dict =  {
            'a1': 1.0, # 1/(uM*sec)
            'a2': 0.5, # 1/(uM*sec)
            'a3': 3.3, # 1/(uM*sec)
            'a4': 10.0, # 1/(uM*sec)
            'a5': 3.3, # 1/(uM*sec)
            'a6': 10.0, # 1/(uM*sec)
            'a7': 20.0, # 1/(uM*sec)
            'a8': 5.0, # 1/(uM*sec)
            'a9': 20.0, # 1/(uM*sec)
            'a10': 5.0, # 1/(uM*sec)
            'd1': 0.4, # 1/sec
            'd2': 0.5, # 1/sec
            'd3': 0.42, # 1/sec
            'd4': 0.8, # 1/sec
            'd5': 0.4, # 1/sec
            'd6': 0.8, # 1/sec
            'd7': 0.6, # 1/sec
            'd8': 0.4, # 1/sec
            'd9': 0.6, # 1/sec
            'd10': 0.4, # 1/sec
            'k1': 0.1, # 1/sec
            'k2': 0.1, # 1/sec
            'k3': 0.1, # 1/sec
            'k4': 0.1, # 1/sec
            'k5': 0.1, # 1/sec
            'k6': 0.1, # 1/sec
            'k7': 0.1, # 1/sec
            'k8': 0.1, # 1/sec
            'k9': 0.1, # 1/sec
            'k10': 0.1, # 1/sec
            'kOn1': 10.0, # 1/(uM*sec)
            'kOn2': 10.0, # 1/(uM*sec)
            'kOff1': 0.05, # 1/sec
            'kOff2': 0.05, # 1/sec
            'kOff3': 0.05, # 1/sec
            'kOff4': 0.5, # 1/sec
            'RAFPase': 0.3, #uM
            'MEKPase': 0.2, #uM
            'MAPKPase': 0.3, #uM
            'total_scaffold': 0.1, #uM -- total scaffold conc (C1 + C2 + ... + C9)
        }

        p_list = [p_dict[key] for key in p_dict.keys()]

        return p_dict, p_list
    


    def get_initial_conditions(self):
        # values come from Ferrell 1996 - trends biochem sci
        #   and Bray and Lay 1997 - PNAS 
        y0_dict = {
            'RAFact': 0.2, #uM
            'RAF': 0.3, # uM 
            'RAF_RAFact': 0.0, # uM 
            'RAFstar': 0.0, # uM 
            'RAFstar_RAFPase': 0.0, # uM 
            'MEK': 0.2, # uM 
            'MEK_RAFstar': 0.0, # uM 
            'MEKstar': 0.0, # uM 
            'MEKstar_MEKPase': 0.0, # uM 
            'MEKstar_RAFstar': 0.0, # uM 
            'MEKstarstar': 0.0, # uM 
            'MEKstarstar_MEKPase': 0.0, # uM 
            'MAPK': 0.4, # uM 
            'MAPK_MEKstarstar': 0.0, # uM 
            'MAPKstar': 0.0, # uM 
            'MAPKstar_MEKstarstar': 0.0, # uM 
            'MAPKstarstar': 0.0, # uM 
            'MAPKstar_MAPKPase': 0.0, # uM 
            'MAPKstarstar_MAPKPase': 0.0, # uM 
            # 'C1': 0.1, #0.0, # uM
            'C2': 0.0, # uM
            'C3': 0.0, # uM
            'C4': 0.0, # uM
            'C5': 0.0, # uM
            'C6': 0.0, # uM
            'C7': 0.0, # uM
            'C8': 0.0, # uM
            'C9': 0.0, # uM
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup