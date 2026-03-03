import equinox as eqx
import jax.numpy as jnp
from jax.lax import cond

class ryu_2015_Rap1(eqx.Module):
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=False):
        self.transient = transient

    def __call__(self, t, y, args):

        # unpack state variables
        EGF, R, R_star, Ras, Ras_star, Raf, Raf_star, MEK, MEK_star, ERK, ERK_star, \
            NFB, NFB_star, PFB, PFB_star, dusp, DUSP, Rap1, Rap1_star = y

        # unpack parameters
        k1, kd1, Ptase_R, ksyn, kdeg, ki, kdi, kdegi, k6, K6, kd6, D6, GAP, k5, \
            K5, kd5, D5, K_NFB, Ptase_Raf, k_PFB, K_PFB, k4, K4, kd4, D4, \
            Ptase_MEK, k2, K2, kd2, D2, k3_F, K3, K3R, kd3, D3, Ptase_NFB, k7, \
            K7, kd7, D7, Ptase_PFB, dusp_basal, dusp_ind, K_dusp, T_dusp, \
            k_Rap1Act, D_Rap1Act, VmaxRap1deact, D_Rap1deact, k_RafRap1, \
            D_RafRap1 = args

        # expressions
        # EGF = 1.0 # input function

        # fluxes
        # receptor activation
        v1 = k1*R*EGF - kd1*Ptase_R*R_star
        ##### Unused in the full model ####
        # v1a = ksyn
        # v1b = kdeg*R
        # v1c = kdeg*R_star
        # v1d = ki*R_star - kdi*Ri
        # v1e = kdegi*Ri
        ###################################
        # Ras activation
        v6 = k6*R_star*(Ras/ (K6 + Ras)) - kd6*GAP*(Ras_star/ (D6 + Ras_star))
        # Raf activation and MAPK cascade
        v5 = k5*Ras_star*(Raf/(K5 + Raf))*(K_NFB**2/(K_NFB**2 + NFB_star**2)) - kd5*Ptase_Raf*(Raf_star/(D5 + Raf_star))
        v5a = k_PFB*PFB_star*(Raf/(K_PFB + Raf))
        # v5b = k_PFB*ERK_star*(Raf/(K_PFB + Raf)) Unused in the full model ####
        v4 = k4*Raf_star*(MEK/(K4 + MEK)) - kd4*Ptase_MEK*(MEK_star/(D4 + MEK_star))
        v2 = k2*MEK_star*(ERK/(K2 + ERK)) - kd2*DUSP*(ERK_star/(D2 + ERK_star))
        # Negative feedback
        v3a = k3_F*ERK_star*(NFB/(K3 + NFB))*(R_star**2/(K3R**2 + R_star**2)) - kd3*Ptase_NFB*(NFB_star/(D3 + NFB_star))
        #### Unused in the full model ####
        # v3b = k3_F*ERK_star*(NFB/(K3 + NFB)) - kd3*Ptase_NFB*(NFB_star/(D3 + NFB_star))
        # Positive feedback
        v7a = k7*ERK_star*R_star*(PFB/(K7 + PFB)) - kd7*Ptase_PFB*(PFB_star/(D7 + PFB_star))
        #### Unused in the full model ####
        # v7b = k7*ERK_star*R_star*(PFB/(K7 + PFB)) - kd7*Ptase_PFB*(PFB_star/(D7 + PFB_star))
        # DUSP expression and slow negative feedback
        v8 = dusp_basal*(1 + dusp_ind*(ERK_star**2/(K_dusp + ERK_star**2)))*(jnp.log10(2)/T_dusp)
        v9 = dusp*(jnp.log10(2)/T_dusp)
        v10 = dusp*(jnp.log10(2)/T_dusp)
        v11 = DUSP*(jnp.log10(2)/T_dusp)
        # Rap1 activation NEW
        v12 = k_Rap1Act*R_star*(Rap1/(D_Rap1Act + Rap1)) - VmaxRap1deact*(Rap1_star/(D_Rap1deact + Rap1_star))
        v13 = k_RafRap1*Rap1_star*(Raf/(D_RafRap1 + Raf))

        # ODE rhs
        trans_fun = lambda k1, R, EGF: jnp.squeeze(-k1*R*EGF)
        sus_fun = lambda k1, R, EGF: 0.0
        d_EGF = cond(self.transient, trans_fun, sus_fun, k1, R, EGF)
        # if self.transient:
        #     d_EGF = -k1*R*EGF
        # else:
        #     d_EGF = 0.0
        d_R = -v1
        d_R_star = v1
        d_Ras = -v6
        d_Ras_star = v6
        d_Raf = -v5 - v5a - v13
        d_Raf_star = v5 + v5a + v13 # v13 is new Rap1 dependent term
        d_MEK = -v4
        d_MEK_star = v4
        d_ERK = -v2
        d_ERK_star = v2
        d_NFB = -v3a
        d_NFB_star = v3a
        d_PFB = -v7a
        d_PFB_star = v7a
        d_dusp = v8 - v9
        d_DUSP = v10 - v11
        d_Rap1 = -v12
        d_Rap1_star = v12 


        # concatenate into tuple and return
        return (d_EGF, d_R, d_R_star, d_Ras, d_Ras_star, d_Raf, d_Raf_star, d_MEK, 
                d_MEK_star, d_ERK, d_ERK_star, d_NFB, d_NFB_star, d_PFB, 
                d_PFB_star, d_dusp, d_DUSP, d_Rap1, d_Rap1_star)


    def get_nominal_params(self):

        p_dict =  {
            'k1':0.5, #1/min
            'kd1':0.5, #1/min
            'Ptase_R':1.0, 
            'ksyn':0.0014, #1/min
            'kdeg':0.0014, #1/min
            'ki':0.22, #1/min
            'kdi':0.044, #1/min
            'kdegi':0.462, #1/min
            'k6':40.0, 
            'K6':1.0,
            'kd6':7.5,
            'D6':1.0,
            'GAP':1.0,
            'k5':10.0,
            'K5':1.0,
            'kd5':3.75,
            'D5':1.0,
            'K_NFB':0.05,
            'Ptase_Raf':1.0,
            'k_PFB':0.0, # set to 0 for EGF and 0.75 for NGF
            'K_PFB':0.01, # 0.01 for v5a and 0.05 for v5b
            'k4':2.0,
            'K4':1.0,
            'kd4':0.5,
            'D4':1.0,
            'Ptase_MEK':1.0,
            'k2':2.0,
            'K2':1.0,
            'kd2':0.25,
            'D2':0.1,
            'k3_F':0.0286,
            'K3':0.01,
            'K3R':0.85,
            'kd3':0.0057,
            'D3':0.5,
            'Ptase_NFB':1.0,
            'k7':0.1, # 0.1 for v7a and 0.04 for v7b/model 2 and 0.01 for v7b/model 3
            'K7':1.0,
            'kd7':0.005,
            'D7':0.1,
            'Ptase_PFB':1.0,
            'dusp_basal':1.0,
            'dusp_ind':6.0,
            'K_dusp':0.1,
            'T_dusp':90.0,
            # new Rap1 parameters
            'k_Rap1Act':0.5, # 1/min
            'D_Rap1Act':1.0,
            'VmaxRap1deact':0.5, # 1/min
            'D_Rap1deact':1.0,
            'k_RafRap1':5.0, # 1/min
            'D_RafRap1':1.0,
        }

        p_list = [p_dict[key] for key in p_dict.keys()]

        return p_dict, p_list



    def get_initial_conditions(self):

        y0_dict = {
            'EGF':1.0,
            'R': 1.0,
            'R_star': 0.0,
            'Ras': 1.0,
            'Ras_star': 0.0,
            'Raf': 1.0,
            'Raf_star': 0.0,
            'MEK': 1.0,
            'MEK_star': 0.0,
            'ERK': 1.0,
            'ERK_star': 0.0,
            'NFB': 1.0,
            'NFB_star': 0.0,
            'PFB': 1.0,
            'PFB_star': 0.0,
            'dusp': 1.0,
            'DUSP': 1.0,
            'Rap1': 1.0,
            'Rap1_star': 0.0,
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup
    