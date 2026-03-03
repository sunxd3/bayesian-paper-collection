import equinox as eqx
from jax.lax import cond
import jax.numpy as jnp

class huang_ferrell_1996(eqx.Module):

    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=True): # defaults to stim from paper
        self.transient = transient

    def __call__(self, t, y, args):
        # unpack state
        E1,MKKK_E1, MKKK_P, MKKK_P_E2, MKK_MKKK_P, MKK_P, MKK_P_MKKPase, MKK_P_MKKK_P, MKK_PP, MKK_PP_MKKPase, MAPK_MKK_PP, MAPK_P, MAPK_P_MAPKPase, MAPK_P_MKK_PP, MAPK_PP,  MAPK_PP_MAPKPase = y
        
        # unnpack parameters
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10, MKKK_tot,MKK_tot,MAPK_tot,E2_tot,MKKPase_tot,MAPKPase_tot = args

        MKKK = MKKK_tot - MKKK_P - MKKK_E1 - MKKK_P_E2 - MKK_MKKK_P - MKK_P_MKKK_P
        MKK = MKK_tot - MKK_P - MKK_PP - MKK_MKKK_P - MKK_P_MKKK_P - MKK_P_MKKPase - MKK_PP_MKKPase - MAPK_MKK_PP - MAPK_P_MKK_PP
        MAPK = MAPK_tot - MAPK_P - MAPK_PP - MAPK_MKK_PP - MAPK_P_MKK_PP - MAPK_P_MAPKPase - MAPK_PP_MAPKPase
        MKKPase = MKKPase_tot - MKK_P_MKKPase - MKK_PP_MKKPase
        MAPKPase = MAPKPase_tot - MAPK_P_MAPKPase + MAPK_PP_MAPKPase
        E2 = E2_tot - MKKK_P_E2
        
        # define fluxes
        J1 = a1*MKKK*E1
        J2 = d1*MKKK_E1
        J3 = k2*MKKK_P_E2
        J4 = k1*MKKK_E1
        J5 = a2*MKKK_P*E2
        J6 = d2*MKKK_P_E2
        J7 = k3*MKK_MKKK_P
        J8 = d3*MKK_MKKK_P
        J9 = a3*MKKK_P*MKK
        J10 = k5*MKK_P_MKKK_P
        J11 = d5*MKK_P_MKKK_P
        J12 = a5*MKK_P*MKKK_P
        J13 = k4*MKK_P_MKKPase
        J14 = a4*MKK_P*MKKPase
        J15 = d4*MKK_P_MKKPase
        J16 = k6*MKK_PP_MKKPase
        J17 = a6*MKK_PP*MKKPase
        J18 = d6*MKK_PP_MKKPase
        J19 = a7*MKK_PP*MAPK
        J20 = d7*MAPK_MKK_PP
        J21 = k7*MAPK_MKK_PP
        J22 = d9*MAPK_P_MKK_PP
        J23 = k9*MAPK_P_MKK_PP
        J24 = a9*MAPK_P*MKK_PP
        J25 = k8*MAPK_P_MAPKPase
        J26 = a8*MAPK_P*MAPKPase
        J27 = d8*MAPK_P_MAPKPase
        J28 = k10*MAPK_PP_MAPKPase
        J29 = a10*MAPK_PP*MAPKPase
        J30 = d10*MAPK_PP_MAPKPase

        # ODE rhs
        trans_fun = lambda J1, J2, J4: jnp.squeeze(J2 + J4 - J1)
        sus_fun = lambda J1, J2, J4: 0.0
        d_E1 = cond(self.transient, trans_fun, sus_fun, J1, J2, J4)

        d_MKKK_E1 = J1 - J2 - J4
        d_MKKK_P = -J5 + J6 + J4 + J7 + J8 - J9 + J10 + J11 - J12
        d_MKKK_P_E2 = J5 - J6 - J3
        d_MKK_MKKK_P = J9 - J8 - J7
        d_MKK_P = -J14 + J15 + J7 + J16 + J11 - J12
        d_MKK_P_MKKPase = J14 - J15 - J13
        d_MKK_P_MKKK_P = J12 - J11 - J10
        d_MKK_PP = J10 - J17 + J18 - J19 + J20 + J21 + J22 + J23 - J24
        d_MKK_PP_MKKPase = J17 - J18 - J16
        d_MAPK_MKK_PP = J19 - J20 - J21
        d_MAPK_P = J21 - J26 + J27 - J24 + J22 + J28
        d_MAPK_P_MAPKPase = J26 - J27 - J25
        d_MAPK_P_MKK_PP = J24 - J22 - J23
        d_MAPK_PP = -J29 + J30 + J23
        d_MAPK_PP_MAPKPase = J29 - J30 - J28

        return (d_E1, d_MKKK_E1, d_MKKK_P, d_MKKK_P_E2, d_MKK_MKKK_P, 
                d_MKK_P, d_MKK_P_MKKPase, d_MKK_P_MKKK_P, d_MKK_PP, 
                d_MKK_PP_MKKPase, d_MAPK_MKK_PP, d_MAPK_P, 
                d_MAPK_P_MAPKPase, d_MAPK_P_MKK_PP, d_MAPK_PP, 
                d_MAPK_PP_MAPKPase)
    

    def get_nominal_params(self): # as defined in the original paper
        Km1 = 0.3 #nM
        Km2 = 0.3 #nM
        Km3 = 0.3 #nM
        Km4 = 0.3 #nM
        Km5 = 0.3 #nM
        Km6 = 0.3 #nM
        Km7 = 0.3 #nM
        Km8 = 0.3 #nM
        Km9 = 0.3 #nM
        Km10 = 0.3 #nM
        a1 = 75.0 #1/s
        a2 = 75.0 #1/s
        a3 = 75.0 #1/s
        a4 = 75.0 #1/s
        a5 = 75.0 #1/s
        a6 = 75.0 #1/s
        a7 = 75.0 #1/s
        a8 = 75.0 #1/s
        a9 = 75.0 #1/s
        a10 = 75.0 #1/s
        d1 = 0.75 #nM/s
        d2 = 0.75 #nM/s
        d3 = 0.75 #nM/s
        d4 = 0.75 #nM/s
        d5 = 0.75 #nM/s
        d6 = 0.75 #nM/s
        d7 = 0.75 #nM/s
        d8 = 0.75 #nM/s
        d9 = 0.75 #nM/s
        d10 = 0.75 #nM/s

        p_dict = {
            # parameters
            'a1': a1, #1/s
            'a2': a2, #1/s
            'a3': a3, #1/s
            'a4': a4, #1/s
            'a5': a5, #1/s
            'a6': a6, #1/s
            'a7': a7, #1/s
            'a8': a8, #1/s
            'a9': a9, #1/s
            'a10': a10, #1/s
            'd1': d1, #nM/s
            'd2': d2, #nM/s
            'd3': d3, #nM/s
            'd4': d4, #nM/s
            'd5': d5, #nM/s
            'd6': d6, #nM/s
            'd7': d7, #nM/s
            'd8': d8, #nM/s
            'd9': d9, #nM/s
            'd10': d10, #nM/s
            'k1': a1*Km1 - d1, #nM/s
            'k2': a2*Km2 - d2, #nM/s
            'k3': a3*Km3 - d3, #nM/s
            'k4': a4*Km4 - d4, #nM/s
            'k5': a5*Km5 - d5, #nM/s
            'k6': a6*Km6 - d6, #nM/s
            'k7': a7*Km7 - d7, #nM/s
            'k8': a8*Km8 - d8, #nM/s
            'k9': a9*Km9 - d9, #nM/s
            'k10': a10*Km10 - d10, #nM/s
            'MKKK_tot': 0.003, #nM
            'MKK_tot': 1.2, #nM
            'MAPK_tot': 1.2, #nM
            'E2_tot': 3e-4, #nM
            'MKKPase_tot': 3e-4, #nM
            'MAPKPase_tot': 0.12, #nM
            # 'E1_tot': 1e-5, #nM THIS IS VARIED OVER A WIDE RANGE AS IT IS THE INPUT!
        }

        p_list = [p_dict[key] for key in p_dict.keys()]

        return p_dict, p_list

    def get_initial_conditions(self):
        ic_dict = {
            'E1':  1e-5, # #nM THIS IS VARIED OVER A WIDE RANGE AS IT
            'MKKK_E1': 0.0,
            'MKKK_P': 0.0,
            'MKKK_P_E2': 0.0,
            'MKK_MKKK_P': 0.0,
            'MKK_P': 0.0,
            'MKK_P_MKKPase': 0.0,
            'MKK_P_MKKK_P': 0.0,
            'MKK_PP': 0.0,
            'MKK_PP_MKKPase': 0.0,
            'MAPK_MKK_PP': 0.0,
            'MAPK_P': 0.0,
            'MAPK_P_MAPKPase': 0.0,
            'MAPK_P_MKK_PP': 0.0,
            'MAPK_PP': 0.0,
            'MAPK_PP_MAPKPase': 0.0,
        }

        ic_tup = tuple([ic_dict[key] for key in ic_dict.keys()])

        return ic_dict, ic_tup
    
    def get_total_ERK(self, params, y0):
        "returns total ERK concentration "
        pdict, _ = self.get_nominal_params()
        ERK_tot_idx = list(pdict.keys()).index('MAPK_tot')
        return params[ERK_tot_idx]