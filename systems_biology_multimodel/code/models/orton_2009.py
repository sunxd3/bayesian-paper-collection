import equinox as eqx
from jax.lax import cond
import jax.numpy as jnp

class orton_2009(eqx.Module):
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=False): # defaults to sustained stim
        self.transient = transient

    def __call__(self, t, y, args):
        # unpack state
        EGF, bRafInactive, bRafActive, Rap1Inactive, Rap1Active, C3GInactive, C3GActive, \
            degradedEGFRReceptor, AktInactive, AktActive, PI3KInactive, PI3KActive, \
            P90RskInactive, P90RskActive, ErkInactive, ErkActive, MekInactive, \
            MekActive, Raf1Inactive, Raf1Active, RasInactive, RasActive, \
            SosInactive, SosActive, freeEGFRReceptor, boundEGFRReceptor = y

        # unpack parameters
        k1_Sos_Deactivation, k1_EGF_Binding_Unbinding, PP2AActive, \
            k2_EGF_Binding_Unbinding, k1_EGFReceptor_Degradation_Free, \
            k1_C3G_Deactivation, k1_EGFReceptor_Degradation, k1_Akt_Deactivation, \
            k1_PI3K_Deactivation, Rap1Gap, Kcat_Erk_Activation, \
            k1_P90Rsk_Deactivation, Kcat_Mek_Deactivation, Kcat_Mek_Activation, \
            Kcat_Raf1_Deactivation, Kcat_Raf1_Activation, Kcat_Ras_Deactivation, \
            Kcat_Ras_Activation, Kcat_Sos_Activation, kcat_bRaf_Activation_Ras, \
            km_bRaf_Activation_Ras, kcat_Mek_Activation_bRaf, \
            kcat_bRaf_Deactivation, kcat_bRaf_Activation, kcat_Rap1_Deactivation, \
            kcat_Rap1_Activation, kcat_C3G_Activation, km_Mek_Activation_bRaf, \
            km_bRaf_Deactivation, km_bRaf_Activation, km_Rap1_Deactivation, \
            km_Rap1_Activation, km_C3G_Activation, Raf1PPtase, \
            km_Raf1_Deactivation_Akt, km_Akt_Activation, km_PI3K_Activation_Ras, \
            km_PI3K_Activation_EGFR, km_Sos_Feedback_Deactivation, \
            km_P90Rsk_Activation, km_Erk_Deactivation, km_Erk_Activation, \
            km_Mek_Deactivation, km_Mek_Activation, Kcat_Raf1_Deactivation_Akt, \
            km_Raf1_Deactivation, km_Raf1_Activation, Kcat_Akt_Activation, \
            km_Ras_Deactivation, km_Ras_Activation, Kcat_PI3K_Activation_Ras, \
            Kcat_PI3K_Activation_EGFR, km_Sos_Activation, \
            Kcat_Sos_Feedback_Deactivation, J_reaction_28, \
            Kcat_P90Rsk_Activation, Kcat_Erk_Deactivation, proEGFRReceptor, \
            RasGapActive = args

        # fluxes
        J_EGF_Binding_Unbinding =  - ((k2_EGF_Binding_Unbinding * boundEGFRReceptor) - (k1_EGF_Binding_Unbinding * freeEGFRReceptor * EGF))
        J_Sos_Activation = (Kcat_Sos_Activation * boundEGFRReceptor * SosInactive / (km_Sos_Activation + SosInactive))
        J_Sos_Deactivation = (k1_Sos_Deactivation * SosActive)
        J_Ras_Activation = (Kcat_Ras_Activation * SosActive * RasInactive / (km_Ras_Activation + RasInactive))
        J_Ras_Deactivation = (Kcat_Ras_Deactivation * RasGapActive * RasActive / (km_Ras_Deactivation + RasActive))
        J_Raf1_Activation = (Kcat_Raf1_Activation * RasActive * Raf1Inactive / (km_Raf1_Activation + Raf1Inactive))
        J_Raf1_Deactivation = (Kcat_Raf1_Deactivation * Raf1PPtase * Raf1Active / (km_Raf1_Deactivation + Raf1Active))
        J_Mek_Activation = (Kcat_Mek_Activation * Raf1Active * MekInactive / (km_Mek_Activation + MekInactive))
        J_Mek_Deactivation = (Kcat_Mek_Deactivation * PP2AActive * MekActive / (km_Mek_Deactivation + MekActive))
        J_Erk_Activation = (Kcat_Erk_Activation * ErkInactive * MekActive / (km_Erk_Activation + ErkInactive))
        J_Erk_Deactivation = (Kcat_Erk_Deactivation * ErkActive * PP2AActive / (km_Erk_Deactivation + ErkActive))
        J_P90Rsk_Activation = (Kcat_P90Rsk_Activation * ErkActive * P90RskInactive / (km_P90Rsk_Activation + P90RskInactive))
        J_P90Rsk_Deactivation = (k1_P90Rsk_Deactivation * P90RskActive)
        J_Sos_Feedback_Deactivation = (Kcat_Sos_Feedback_Deactivation * P90RskActive * SosActive / (km_Sos_Feedback_Deactivation + SosActive))
        J_PI3K_Activation_EGFR = (Kcat_PI3K_Activation_EGFR * boundEGFRReceptor * PI3KInactive / (km_PI3K_Activation_EGFR + PI3KInactive))
        J_PI3K_Activation_Ras = (Kcat_PI3K_Activation_Ras * PI3KInactive * RasActive / (km_PI3K_Activation_Ras + PI3KInactive))
        J_PI3K_Deactivation = (k1_PI3K_Deactivation * PI3KActive)
        J_Akt_Activation = (Kcat_Akt_Activation * PI3KActive * AktInactive / (km_Akt_Activation + AktInactive))
        J_Akt_Deactivation = (k1_Akt_Deactivation * AktActive)
        J_Raf1_Deactivation_Akt = (Kcat_Raf1_Deactivation_Akt * AktActive * Raf1Active / (km_Raf1_Deactivation_Akt + Raf1Active))
        J_EGFReceptor_Degradation = (k1_EGFReceptor_Degradation * boundEGFRReceptor)
        J_C3G_Activation = (kcat_C3G_Activation * boundEGFRReceptor * C3GInactive / (km_C3G_Activation + C3GInactive))
        J_C3G_Deactivation = (k1_C3G_Deactivation * C3GActive)
        J_Rap1_Activation = (kcat_Rap1_Activation * C3GActive * Rap1Inactive / (km_Rap1_Activation + Rap1Inactive))
        J_Rap1_Deactivation = (kcat_Rap1_Deactivation * Rap1Active * Rap1Gap / (km_Rap1_Deactivation + Rap1Active))
        J_bRaf_Activation = (kcat_bRaf_Activation * Rap1Active * bRafInactive / (km_bRaf_Activation + bRafInactive))
        J_bRaf_Deactivation = (kcat_bRaf_Deactivation * bRafActive * Raf1PPtase / (km_bRaf_Deactivation + bRafActive))
        J_Mek_Activation_bRaf = (kcat_Mek_Activation_bRaf * bRafActive * MekInactive / (km_Mek_Activation_bRaf + MekInactive))
        J_EGFReceptor_Degradation_Free = (k1_EGFReceptor_Degradation_Free * freeEGFRReceptor)
        J_bRaf_Activation_Ras = (kcat_bRaf_Activation_Ras * bRafInactive * RasActive / (km_bRaf_Activation_Ras + bRafInactive))

        # ODE rhs
        trans_fun = lambda J_EGF_Binding_Unbinding: jnp.squeeze(-J_EGF_Binding_Unbinding)
        sus_fun = lambda J_EGF_Binding_Unbinding: 0.0
        d_EGF = cond(self.transient, trans_fun, sus_fun, J_EGF_Binding_Unbinding)
        # if self.transient:
        #     d_EGF = -J_EGF_Binding_Unbinding
        # else:
        #     d_EGF = 0.0
        d_bRafInactive = ( - J_bRaf_Activation + J_bRaf_Deactivation - J_bRaf_Activation_Ras)
        d_bRafActive = (J_bRaf_Activation - J_bRaf_Deactivation + J_bRaf_Activation_Ras)
        d_Rap1Inactive = ( - J_Rap1_Activation + J_Rap1_Deactivation)
        d_Rap1Active = (J_Rap1_Activation - J_Rap1_Deactivation)
        d_C3GInactive = ( - J_C3G_Activation + J_C3G_Deactivation)
        d_C3GActive = (J_C3G_Activation - J_C3G_Deactivation)
        d_degradedEGFRReceptor = (J_EGFReceptor_Degradation + J_EGFReceptor_Degradation_Free)
        d_AktInactive = ( - J_Akt_Activation + J_Akt_Deactivation)
        d_AktActive = (J_Akt_Activation - J_Akt_Deactivation)
        d_PI3KInactive = ( - J_PI3K_Activation_EGFR - J_PI3K_Activation_Ras + J_PI3K_Deactivation)
        d_PI3KActive = (J_PI3K_Activation_EGFR + J_PI3K_Activation_Ras - J_PI3K_Deactivation)
        d_P90RskInactive = ( - J_P90Rsk_Activation + J_P90Rsk_Deactivation)
        d_P90RskActive = (J_P90Rsk_Activation - J_P90Rsk_Deactivation)
        d_ErkInactive = ( - J_Erk_Activation + J_Erk_Deactivation)
        d_ErkActive = (J_Erk_Activation - J_Erk_Deactivation)
        d_MekInactive = ( - J_Mek_Activation + J_Mek_Deactivation - J_Mek_Activation_bRaf)
        d_MekActive = (J_Mek_Activation - J_Mek_Deactivation + J_Mek_Activation_bRaf)
        d_Raf1Inactive = ( - J_Raf1_Activation + J_Raf1_Deactivation + J_Raf1_Deactivation_Akt)
        d_Raf1Active = (J_Raf1_Activation - J_Raf1_Deactivation - J_Raf1_Deactivation_Akt)
        d_RasInactive = ( - J_Ras_Activation + J_Ras_Deactivation)
        d_RasActive = (J_Ras_Activation - J_Ras_Deactivation)
        d_SosInactive = ( - J_Sos_Activation + J_Sos_Deactivation + J_Sos_Feedback_Deactivation)
        d_SosActive = (J_Sos_Activation - J_Sos_Deactivation - J_Sos_Feedback_Deactivation)
        d_freeEGFRReceptor = ( - J_EGF_Binding_Unbinding + J_reaction_28 - J_EGFReceptor_Degradation_Free)
        d_boundEGFRReceptor = (J_EGF_Binding_Unbinding - J_EGFReceptor_Degradation)

        # concatenate into tuple and return
        return (d_EGF, d_bRafInactive, d_bRafActive, d_Rap1Inactive, d_Rap1Active,
            d_C3GInactive, d_C3GActive, d_degradedEGFRReceptor, d_AktInactive,
            d_AktActive, d_PI3KInactive, d_PI3KActive, d_P90RskInactive,
            d_P90RskActive, d_ErkInactive, d_ErkActive, d_MekInactive,
            d_MekActive, d_Raf1Inactive, d_Raf1Active, d_RasInactive,
            d_RasActive, d_SosInactive, d_SosActive, d_freeEGFRReceptor,
            d_boundEGFRReceptor)
    

    def get_nominal_params(self):
        
        p_dict =  {
            'k1_Sos_Deactivation': 2.5,
            'k1_EGF_Binding_Unbinding': 2.18503e-5,
            'PP2AActive': 120000.0,
            'k2_EGF_Binding_Unbinding': 0.0121008,
            'k1_EGFReceptor_Degradation_Free': 0.00125,
            'k1_C3G_Deactivation': 2.5,
            'k1_EGFReceptor_Degradation': 0.2,
            'k1_Akt_Deactivation': 0.005,
            'k1_PI3K_Deactivation': 0.005,
            'Rap1Gap': 120000.0,
            'Kcat_Erk_Activation': 9.85367,
            'k1_P90Rsk_Deactivation': 0.005,
            'Kcat_Mek_Deactivation': 2.83243,
            'Kcat_Mek_Activation': 185.759,
            'Kcat_Raf1_Deactivation': 0.126329,
            'Kcat_Raf1_Activation': 0.884096,
            'Kcat_Ras_Deactivation': 1509.36,
            'Kcat_Ras_Activation': 32.344,
            'Kcat_Sos_Activation': 694.731,
            'kcat_bRaf_Activation_Ras': 0.884096,
            'km_bRaf_Activation_Ras': 62464.6,
            'kcat_Mek_Activation_bRaf': 185.759,
            'kcat_bRaf_Deactivation': 0.126329,
            'kcat_bRaf_Activation': 0.884096,
            'kcat_Rap1_Deactivation': 1509.36,
            'kcat_Rap1_Activation': 32.344,
            'kcat_C3G_Activation': 694.731,
            'km_Mek_Activation_bRaf': 4768350.0,
            'km_bRaf_Deactivation': 1061.71,
            'km_bRaf_Activation': 62464.6,
            'km_Rap1_Deactivation': 1432410.0,
            'km_Rap1_Activation': 35954.3,
            'km_C3G_Activation': 6086070.0,
            'Raf1PPtase': 120000.0,
            'km_Raf1_Deactivation_Akt': 119355.0,
            'km_Akt_Activation': 653951.0,
            'km_PI3K_Activation_Ras': 272056.0,
            'km_PI3K_Activation_EGFR': 184912.0,
            'km_Sos_Feedback_Deactivation': 896896.0,
            'km_P90Rsk_Activation': 763523.0,
            'km_Erk_Deactivation': 3496490.0,
            'km_Erk_Activation': 1007340.0,
            'km_Mek_Deactivation': 518753.0,
            'km_Mek_Activation': 4768350.0,
            'Kcat_Raf1_Deactivation_Akt': 15.1212,
            'km_Raf1_Deactivation': 1061.71,
            'km_Raf1_Activation': 62464.6,
            'Kcat_Akt_Activation': 0.0566279,
            'km_Ras_Deactivation': 1432410.0,
            'km_Ras_Activation': 35954.3,
            'Kcat_PI3K_Activation_Ras': 0.0771067,
            'Kcat_PI3K_Activation_EGFR': 10.6737,
            'km_Sos_Activation': 6086070.0,
            'Kcat_Sos_Feedback_Deactivation': 1611.97,
            'J_reaction_28': 100.0,
            'Kcat_P90Rsk_Activation': 0.0213697,
            'Kcat_Erk_Deactivation': 8.8912,
            'proEGFRReceptor': 1.0,
            'RasGapActive': 120000.0,
        }

        p_list = [p_dict[key] for key in p_dict.keys()]

        return p_dict, p_list
    


    def get_initial_conditions(self):
        
        y0_dict = {
            'EGF': 1.0002e7,
            'bRafInactive':120000.0,
            'bRafActive':0.0,
            'Rap1Inactive':120000.0,
            'Rap1Active':0.0,
            'C3GInactive':120000.0,
            'C3GActive':0.0,
            'degradedEGFRReceptor':0.0,
            'AktInactive':120000.0,
            'AktActive':0.0,
            'PI3KInactive':120000.0,
            'PI3KActive':0.0,
            'P90RskInactive':120000.0,
            'P90RskActive':0.0,
            'ErkInactive':600000.0,
            'ErkActive':0.0,
            'MekInactive':600000.0,
            'MekActive':0.0,
            'Raf1Inactive':120000.0,
            'Raf1Active':0.0,
            'RasInactive':120000.0,
            'RasActive':0.0,
            'SosInactive':120000.0,
            'SosActive':0.0,
            'freeEGFRReceptor':80000.0,
            'boundEGFRReceptor':0.0,     
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup
    

