import equinox as eqx
import jax.numpy as jnp
from jax.lax import cond

class kochanczyk_2017(eqx.Module):
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=False):
        self.transient = transient

    def __call__(self, t, y, args):

        # unpack state variables
        EGF, EGFRegfIsos, RASgGDPsos, RASgGTPsos, SOSSUegfrrem, RasGAPras, RAFSI, \
            MEKSUT292U, ERKSU, EKAREVactI, ERKTRactI, EGFRegfAsos, \
            RASgGTP1sos_RasGAPras1, RAFSA, EGFRegfAsos1_SOSSUegfr1rem, \
            RASgGDP1sos_RasGAPras1, MEKSPT292U, EGFRegfIsos1_SOSSUegfr1rem, \
            EGFRegfAsos1_RASgGTPsos2_SOSSUegfr1rem2, \
            EGFRegfAsos1_RASgGDPsos2_SOSSUegfr1rem2, MEKSPPT292U, \
            EGFRegfIsos1_RASgGTPsos2_SOSSUegfr1rem2, \
            EGFRegfIsos1_RASgGDPsos2_SOSSUegfr1rem2, ERKSP, ERKSPP, \
            SOSSPegfrrem, MEKSUT292P, MEKSPT292P, MEKSPPT292P, RAFSP, \
            EKAREVactA, ERKTRactA, SOSSPPegfrrem, SOSSPPPegfrrem, SOSSPPPPegfrrem = y
        
        # unpack parameters
        ERKpp_SOS1_FB, ERKpp_MEK_FB, ERKpp_RAF1_FB, BIMOL, RAS_t0_active, \
            d1, u1a, u1b, u2a, u2b, u3, k3, d2, q1, q2, q3, q4, q5, q6, \
            a0_ekarev, d0_ekarev, a0_erktr, d0_erktr = args

        # expressions
       
        # fluxes
        v1 = (5e-5*EGF)*EGFRegfIsos
        v2 = (1e-5/BIMOL)*RasGAPras*RASgGTPsos
        v3 = (1e-7/BIMOL)*RASgGTPsos*RAFSI
        v4 = d1*EGFRegfAsos
        v5 = (1e-5/BIMOL)*EGFRegfAsos*SOSSUegfrrem
        v6 = k3*RASgGTP1sos_RasGAPras1
        v7 = d2*RAFSA
        v8 = ((1e-7/BIMOL)*2)*RAFSA*MEKSUT292U
        v9 = d1*EGFRegfAsos1_SOSSUegfr1rem
        v10 = u1a*EGFRegfAsos1_SOSSUegfr1rem
        v11 = (1e-6/BIMOL)*EGFRegfAsos1_SOSSUegfr1rem*RASgGTPsos
        v12 = (1e-7/BIMOL)*EGFRegfAsos1_SOSSUegfr1rem*RASgGDPsos
        v13 = u3*RASgGDP1sos_RasGAPras1
        v14 = (1e-7/BIMOL)*RAFSA*MEKSPT292U
        v15 = q1*MEKSPT292U
        v16 = (5e-5*EGF)*EGFRegfIsos1_SOSSUegfr1rem
        v17 = d1*EGFRegfAsos1_RASgGTPsos2_SOSSUegfr1rem2
        v18 = d1*EGFRegfAsos1_RASgGDPsos2_SOSSUegfr1rem2
        v19 = u1b*EGFRegfIsos1_SOSSUegfr1rem
        v20 = (1e-6/BIMOL)*EGFRegfIsos1_SOSSUegfr1rem*RASgGTPsos
        v21 = u2a*EGFRegfAsos1_RASgGTPsos2_SOSSUegfr1rem2
        v22 = (1e-7/BIMOL)*EGFRegfIsos1_SOSSUegfr1rem*RASgGDPsos
        v23 = u2b*EGFRegfAsos1_RASgGDPsos2_SOSSUegfr1rem2
        v24 = (1e-4/BIMOL)*EGFRegfAsos1_RASgGTPsos2_SOSSUegfr1rem2*RASgGDPsos
        v25 = (1e-5/BIMOL)*EGFRegfAsos1_RASgGDPsos2_SOSSUegfr1rem2*RASgGDPsos
        v26 = (q1*2)*MEKSPPT292U
        v27 = ((3e-6/BIMOL)*2)*MEKSPPT292U*ERKSU
        v28 = (5e-5*EGF)*EGFRegfIsos1_RASgGTPsos2_SOSSUegfr1rem2
        v29 = (5e-5*EGF)*EGFRegfIsos1_RASgGDPsos2_SOSSUegfr1rem2
        v30 = u2a*EGFRegfIsos1_RASgGTPsos2_SOSSUegfr1rem2
        v31 = u2b*EGFRegfIsos1_RASgGDPsos2_SOSSUegfr1rem2
        v32 = (1e-4/BIMOL)*EGFRegfIsos1_RASgGTPsos2_SOSSUegfr1rem2*RASgGDPsos
        v33 = (1e-5/BIMOL)*EGFRegfIsos1_RASgGDPsos2_SOSSUegfr1rem2*RASgGDPsos
        v34 = (3e-6/BIMOL)*MEKSPPT292U*ERKSP
        v35 = q2*ERKSP
        v36 = (q2*2)*ERKSPP
        v37 = (((3e-9/BIMOL)*ERKpp_SOS1_FB)*4)*ERKSPP*SOSSUegfrrem
        v38 = ((6e-10/BIMOL)*ERKpp_MEK_FB)*ERKSPP*MEKSUT292U
        v39 = ((6e-10/BIMOL)*ERKpp_MEK_FB)*ERKSPP*MEKSPT292U
        v40 = ((6e-10/BIMOL)*ERKpp_MEK_FB)*ERKSPP*MEKSPPT292U
        v41 = ((6e-10/BIMOL)*ERKpp_RAF1_FB)*ERKSPP*RAFSA
        v42 = ((6e-10/BIMOL)*ERKpp_RAF1_FB)*ERKSPP*RAFSI
        v43 = a0_ekarev*ERKSPP*EKAREVactI
        v44 = a0_erktr*ERKSPP*ERKTRactI
        v45 = ((1e-7/BIMOL)*2)*RAFSA*MEKSUT292P
        v46 = (1e-7/BIMOL)*RAFSA*MEKSPT292P
        v47 = (q1*2)*MEKSPPT292P
        v48 = q1*MEKSPT292P
        v49 = ((3e-6/BIMOL)*2)*MEKSPPT292P*ERKSU
        v50 = (3e-6/BIMOL)*MEKSPPT292P*ERKSP
        v51 = (((3e-9/BIMOL)*ERKpp_SOS1_FB)*3)*ERKSPP*SOSSPegfrrem
        v52 = q3*SOSSPegfrrem
        v53 = q4*MEKSUT292P
        v54 = q4*MEKSPT292P
        v55 = q4*MEKSPPT292P
        v56 = (q5*2)*MEKSPPT292P
        v57 = q5*MEKSPT292P
        v58 = q6*RAFSP
        v59 = d0_ekarev*EKAREVactA
        v60 = d0_erktr*ERKTRactA
        v61 = (((3e-9/BIMOL)*ERKpp_SOS1_FB)*2)*ERKSPP*SOSSPPegfrrem
        v62 = (q3*2)*SOSSPPegfrrem
        v63 = ((3e-9/BIMOL)*ERKpp_SOS1_FB)*ERKSPP*SOSSPPPegfrrem
        v64 = (q3*3)*SOSSPPPegfrrem
        v65 = (q3*4)*SOSSPPPPegfrrem
    

        # ODE rhs
        trans_fun = lambda v1: jnp.squeeze(-v1)
        sus_fun = lambda v1: 0.0
        d_EGF = cond(self.transient, trans_fun, sus_fun, v1)
        # if self.transient:
        #     d_EGF = -v1
        # else:
        #     d_EGF = 0.0
        d_EGFRegfIsos = -v1 +v4 +v19
        d_RASgGDPsos = -v12 +v13 -v22 +v23 -v24 -v25 +v31 -v32 -v33
        d_RASgGTPsos = -v2 -v11 -v20 +v21 +v24 +v25 +v30 +v32 +v33
        d_SOSSUegfrrem = -v5 +v10 +v19 -v37 +v52
        d_RasGAPras = -v2 +v13
        d_RAFSI = -v3 +v7 -v42 +v58
        d_MEKSUT292U = -v8 +v15 -v38 +v53
        d_ERKSU = -v27 +v35 -v49
        d_EKAREVactI = -v43 +v59
        d_ERKTRactI = -v44 +v60
        d_EGFRegfAsos = v1 -v4 -v5 +v10
        d_RASgGTP1sos_RasGAPras1 = v2 -v6
        d_RAFSA = v3 -v7 -v41
        d_EGFRegfAsos1_SOSSUegfr1rem = v5 -v9 -v10 -v11 -v12 +v16 +v21 +v23
        d_RASgGDP1sos_RasGAPras1 = v6 -v13
        d_MEKSPT292U = v8 -v14 -v15 +v26 -v39 +v54
        d_EGFRegfIsos1_SOSSUegfr1rem = v9 -v16 -v19 -v20 -v22 +v30 +v31
        d_EGFRegfAsos1_RASgGTPsos2_SOSSUegfr1rem2 = v11 -v17 -v21 +v28
        d_EGFRegfAsos1_RASgGDPsos2_SOSSUegfr1rem2 = v12 -v18 -v23 +v29
        d_MEKSPPT292U = v14 -v26 -v40 +v55
        d_EGFRegfIsos1_RASgGTPsos2_SOSSUegfr1rem2 = v17 +v20 -v28 -v30
        d_EGFRegfIsos1_RASgGDPsos2_SOSSUegfr1rem2 = v18 +v22 -v29 -v31
        d_ERKSP = v27 -v34 -v35 +v36 +v49 -v50
        d_ERKSPP = v34 -v36 +v50
        d_SOSSPegfrrem = v37 -v51 -v52 +v62
        d_MEKSUT292P = v38 -v45 +v48 -v53 +v57
        d_MEKSPT292P = v39 +v45 -v46 +v47 -v48 -v54 +v56 -v57
        d_MEKSPPT292P = v40 +v46 -v47 -v55 -v56
        d_RAFSP = v41 +v42 -v58
        d_EKAREVactA = v43 -v59
        d_ERKTRactA = v44 -v60
        d_SOSSPPegfrrem = v51 -v61 -v62 +v64
        d_SOSSPPPegfrrem = v61 -v63 -v64 +v65
        d_SOSSPPPPegfrrem = v63 -v65


        # concatenate into tuple and return
        return (d_EGF, d_EGFRegfIsos, d_RASgGDPsos, d_RASgGTPsos, d_SOSSUegfrrem, 
                d_RasGAPras, d_RAFSI, d_MEKSUT292U, d_ERKSU, d_EKAREVactI, 
                d_ERKTRactI, d_EGFRegfAsos, d_RASgGTP1sos_RasGAPras1, d_RAFSA, 
                d_EGFRegfAsos1_SOSSUegfr1rem, d_RASgGDP1sos_RasGAPras1, 
                d_MEKSPT292U, d_EGFRegfIsos1_SOSSUegfr1rem, 
                d_EGFRegfAsos1_RASgGTPsos2_SOSSUegfr1rem2, 
                d_EGFRegfAsos1_RASgGDPsos2_SOSSUegfr1rem2, d_MEKSPPT292U,
                d_EGFRegfIsos1_RASgGTPsos2_SOSSUegfr1rem2, 
                d_EGFRegfIsos1_RASgGDPsos2_SOSSUegfr1rem2, d_ERKSP, d_ERKSPP, 
                d_SOSSPegfrrem, d_MEKSUT292P, d_MEKSPT292P, d_MEKSPPT292P, 
                d_RAFSP, d_EKAREVactA, d_ERKTRactA, d_SOSSPPegfrrem, 
                d_SOSSPPPegfrrem, d_SOSSPPPPegfrrem,)

    def get_nominal_params(self):

        p_dict =  {
            'ERKpp_SOS1_FB':1,
            'ERKpp_MEK_FB':1,
            'ERKpp_RAF1_FB':1,
            'BIMOL':1,
            'RAS_t0_active':0,
            'd1':0.01,
            'u1a':0.01,
            'u1b':100,
            'u2a':1,
            'u2b':1,
            'u3':0.01,
            'k3':100,
            'd2':0.01,
            'q1':0.01,
            'q2':0.01,
            'q3':0.0003,
            'q4':0.0003,
            'q5':100,
            'q6':0.0003,
            'a0_ekarev':3e-09,
            'd0_ekarev':0.002,
            'a0_erktr':1e-09,
            'd0_erktr':0.001,
        }

        p_list = [p_dict[key] for key in p_dict.keys()]

        return p_dict, p_list



    def get_initial_conditions(self):

        y0_dict = {
            'EGF':10.0,
            'EGFRegfIsos':300000.0,
            'RASgGDPsos':60000.0,
            'RASgGTPsos':0.0,
            'SOSSUegfrrem':100000.0,
            'RasGAPras':6000.0,
            'RAFSI':500000.0,
            'MEKSUT292U':200000.0,
            'ERKSU':3000000.0,
            'EKAREVactI':1000000.0,
            'ERKTRactI':1000000.0,
            'EGFRegfAsos':0.0,
            'RASgGTP1sos_RasGAPras1':0.0,
            'RAFSA':0.0,
            'EGFRegfAsos1_SOSSUegfr1rem':0.0,
            'RASgGDP1sos_RasGAPras1':0.0,
            'MEKSPT292U':0.0,
            'EGFRegfIsos1_SOSSUegfr1rem':0.0,
            'EGFRegfAsos1_RASgGTPsos2_SOSSUegfr1rem2':0.0,
            'EGFRegfAsos1_RASgGDPsos2_SOSSUegfr1rem2':0.0,
            'MEKSPPT292U':0.0,
            'EGFRegfIsos1_RASgGTPsos2_SOSSUegfr1rem2':0.0,
            'EGFRegfIsos1_RASgGDPsos2_SOSSUegfr1rem2':0.0,
            'ERKSP':0.0,
            'ERKSPP':0.0,
            'SOSSPegfrrem':0.0,
            'MEKSUT292P':0.0,
            'MEKSPT292P':0.0,
            'MEKSPPT292P':0.0,
            'RAFSP':0.0,
            'EKAREVactA':0.0,
            'ERKTRactA':0.0,
            'SOSSPPegfrrem':0.0,
            'SOSSPPPegfrrem':0.0,
            'SOSSPPPPegfrrem':0.0,
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup
    


