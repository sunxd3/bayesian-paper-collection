import equinox as eqx
import jax.numpy as jnp
from jax.lax import cond

class shin_2014_Rap1(eqx.Module):
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=False):
        self.transient = transient

    def __call__(self, t, y, args):
        # unpack state
        EGF, RE, GS, Ras_GTP, act_Raf, pp_MEK, pp_ERK, Rap1, Rap1_act = y

        # unpack parameters
        EGFR_tot, SOS_tot, Grb2_tot, Ras_tot, Raf_tot, MEK_tot, ERK_tot, ka38, \
            kd38, ka39a, ki39, kd39, kc40, kc41, kc42, kc43, kc44, kc45, kc46, \
            kc47, alpha, kRap1Act, kRap1deAct, kRap1_RafAct = args

        # define algebraic equations
        EGFR = EGFR_tot - RE
        SOS = SOS_tot - GS
        Grb2 = Grb2_tot
        Ras_GDP = Ras_tot - Ras_GTP
        Raf = Raf_tot - act_Raf
        MEK = MEK_tot - pp_MEK
        ERK = ERK_tot - pp_ERK
        # Rap1 = Rap1_tot - Rap1_act # new Rap1

        # ode
        # EGF
        trans_fun = lambda ka38, EGF, EGFR: jnp.squeeze(-ka38*EGF*EGFR)
        sus_fun = lambda ka38, EGF, EGFR: 0.0
        d_EGF_dt = cond(self.transient, trans_fun, sus_fun, ka38, EGF, EGFR)
        # if self.transient == True:
        #     d_EGF_dt = -ka38*EGF*EGFR
        # else:
        #     d_EGF_dt = 0.0
        # RE
        d_RE_dt = ka38*EGF*EGFR - kd38*RE
        # GS
        d_GS_dt = ka39a*RE*SOS*Grb2/(1 + alpha*((pp_ERK/ki39)**3)) - kd39*GS
        # Ras_GTP
        d_Ras_GTP_dt = kc40*GS*Ras_GDP - kc41*Ras_GTP
        # act_Raf
        # new Rap1_act dependent term
        d_act_Raf_dt = (kc42*Ras_GTP + kRap1_RafAct*Rap1_act)*Raf - kc43*act_Raf
        # pp_MEK
        d_pp_MEK_dt = kc44*act_Raf*MEK - kc45*pp_MEK
        # pp_ERK
        d_pp_ERK_dt = kc46*pp_MEK*ERK - kc47*pp_ERK
        # activated Rap1
        d_Rap1_dt = - kRap1Act*RE*Rap1 + kRap1deAct*Rap1_act
        d_Rap1_act_dt = kRap1Act*RE*Rap1 - kRap1deAct*Rap1_act # new Rap1

        # return jnp.array((d_EGF_dt, d_RE_dt, d_GS_dt, d_Ras_GTP_dt, d_act_Raf_dt, d_pp_MEK_dt, d_pp_ERK_dt))
        return (d_EGF_dt, d_RE_dt, d_GS_dt, d_Ras_GTP_dt, d_act_Raf_dt, \
                d_pp_MEK_dt, d_pp_ERK_dt, d_Rap1_dt, d_Rap1_act_dt)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        y0_dict = {'EGF': 1.0e-2,
            'RE': 0.0, # uM
            'GS': 0.0, # uM
            'Ras_GTP': 0.0, # uM
            'act_Raf': 0.0, # uM
            'pp_MEK': 0.0, # uM
            'pp_ERK': 0.0, # uM
            'Rap1': 1e-1, # uM
            'Rap1_act': 0.0, # uM
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup

        
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        param_dict = {
            'EGFR_tot': 1.99e-3, # uM
            'SOS_tot': 1.18e-2, # uM
            'Grb2_tot': 7.74e02, # uM
            'Ras_tot': 1.25e-1, # uM
            'Raf_tot': 2.17e-1, # uM
            'MEK_tot': 6.74e-2, # uM
            'ERK_tot': 2.26e-1, # uM
            'ka38': 3.36e1, # 1/(uM*min)
            'kd38': 6.925e3, # 1/min
            'ka39a': 1.088e5, # 1/(uM^2*min)
            'ki39': 7.716e-4, # uM
            'kd39': 1.616e-1, # 1/min
            'kc40': 4.869e2, # 1/(uM*min)
            'kc41': 5.073e1, # 1/min
            'kc42': 4.648e2, # 1/(uM*min)
            'kc43': 4.663, # 1/min
            'kc44': 4.037e2, # 1/(uM*min)
            'kc45': 6.872e-2, # 1/min
            'kc46': 7.821, # 1/(uM*min)
            'kc47': 3.905e-1, # 1/min
            # new params
            'alpha': 1.0, # unitless controls the strength of the feedback (set to 0 or 1)
            'kRap1Act': 1.0, # 1/min
            'kRap1deAct': 1.0, # 1/min
            'kRap1_RafAct': 10.0, # 1/min
        }
        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

    def get_total_ERK(self, params, y0):
        "returns total ERK concentration "
        pdict, _ = self.get_nominal_params()
        ERK_tot_idx = list(pdict.keys()).index('ERK_tot')
        return params[ERK_tot_idx]