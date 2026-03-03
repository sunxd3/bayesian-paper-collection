import equinox as eqx
from jax.lax import cond
import jax.numpy as jnp

class vonKriegsheim_2009_Rap1(eqx.Module):
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=False): # defaults to sustained stim
        self.transient = transient

    def __call__(self, t, y, args):
        # unpack state variables
        EGF, EGFR, pEGFR, pEGFRi, PIP3, Akt, pAkt, \
            PLCg, pPLCg, PKC, PKCa, RKIP, pRKIP, RasD, RasT, NF1, pNF1, Raf, \
            Rafa, MEK, ppMEK, ERK, ppERK, ERK_15, ppERK_15, ppERKn, PEA, \
            p104PEA, p116PEA, EGFRi, RSK, pRSK, Rap1, Rap1_act = y
        
        # commented on 3/19/24 to exclude NGF/NGFR from the model
        # NGF, EGF, NGFR, pNGFR, EGFR, pEGFR, pEGFRi, PIP3, Akt, pAkt, PhE, \
            # PLCg, pPLCg, PKC, PKCa, RKIP, pRKIP, RasD, RasT, NF1, pNF1, Raf, \
            # Rafa, MEK, ppMEK, ERK, ppERK, ERK_15, ppERK_15, ppERKn, PEA, \
            # p104PEA, p116PEA, EGFRi, NGFRi, RSK, pRSK = y

        # unpack parameters
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, \
        k17, k18, k19,  k20,  k21,  k22,  k23,  k24,  k25,  k26,  k27,  \
        k28,  k29,  k30,  k31,  k32,  k33,  k34,  k35,  k36,  k37, k38, k39,\
        k40, k41, k42, k43, k44, k45, k46, k47, k48, k49,  k50,  k51,  k52, \
        k53, k54, k55, k56, k57, k58, k59, k60, k61, k62, k63, k64, k65, \
        k66, k67, k68,  k69,  k70,  k71,  k72,  k73, V_NUC, V_CYT, \
        NGFDeriv, EGFDeriv, k_Rap1act, k_Rap1deact, k_RafRap1, Kd_RafRap1 = args

        # exclude NGF/NGFR from the model (added 3/19/24)
        NGF = 0.0
        NGFR = 0.0
        pNGFR = 0.0
        PhE = 0.0
        NGFRi = 0.0

        # expressions
        ERKFeedtoSOS=(1+(pRSK)/k49)
        RKIPNegFeed=(k50 + MEK + (k50*RKIP/k51))

        # parameters used for knowdown/inhibition
        # set to 1 for no knockdown/inhibition
        MG=1.0
        GO=1.0
        LY=1.0
        UO=1.0
        ERKDN=1.0
        S116A=1.0
        S104A=1.0

        # fluxes
        #NGFR Activation
        v1=k1*NGF*NGFR
        #NGFR Deactivation
        v2=k2*pNGFR
        #EGFR Activation
        v3=k3*EGF*EGFR
        #EGFR Deactivation
        v4=k4*pEGFR
        #Constitutive EGFR Internalization
        v5=k5*pEGFR
        #EGFR Recycling
        v6=k6*EGFRi
        #EGFR Degradation
        v7=k7*pEGFRi*MG
        #pNGFR PIP3 Synthesis
        v8=k8*pNGFR*LY
        #pEGFR PIP3 Synthesis
        v9=k9*pEGFR*LY
        #PIP3 Destruction
        v10=k10*PIP3
        #pNGFR PLCg Activation
        v11=k11*pNGFR*PLCg
        #pEGFR PLCg Activation
        v12=k12*pEGFR*PLCg
        #PLCg Deactivation
        v13=k13*pPLCg
        #PLCg PKC Activation
        v14=k14*(pPLCg**k61/(k62**k61+pPLCg**k61))*PKC + k63*(pPLCg**k64/(k65**k64+pPLCg**k64))*PKC
        #Phorbol Ester (PhE) PKC Activation
        v15=k15*(PhE**k66/(k67**k66+PhE**k66))*PKC
        #PKC Deactivation
        v16=k16*PKCa+k59*PKCa*EGF+k60*PKCa*NGF
        #RKIP Phosphorylation
        v17=k17*RKIP*PKCa*GO
        #RKIP Dephosphorylation
        v18=k18*pRKIP
        #Surface EGFR Ras Activation
        v19=k19*pEGFR*RasD/ERKFeedtoSOS
        #Surface NGFR Ras Activation
        v20=k20*pNGFR*RasD/ERKFeedtoSOS
        #Internalized EGFR Ras Activation
        v21=k21*pEGFRi*RasD/ERKFeedtoSOS
        #PIP3 Ras Activation
        v22=k22*PIP3*RasD/ERKFeedtoSOS
        #Basal Ras Activation
        v23=k23*RasD
        #PhE Ras Activation
        v24=k24*PhE*RasD
        #PLCg Ras Activation
        v25=k25*(pPLCg**k70/(pPLCg**k70+k71**k70))*RasD
        #Ras Deactivation
        v26=k26*NF1*RasT
        #NF1 Phosphorylation
        v27=k27*ppERK*NF1*ERKDN
        #NF1 Dephosphorylation
        v28=k28*pNF1
        #Raf Activation
        v29=k29*(RasT**k72/(RasT**k72+k73**k72))*Raf
        #Raf Deactivation
        v30=k30*Rafa
        #MEK Activation
        v31=k31*Rafa*MEK/RKIPNegFeed
        #MEK Deactivation
        v32=k32*ppMEK
        #ERK Activation
        v33=k33*ppMEK*ERK*UO
        #ERK Deactivation
        v34=k34*ppERK
        #ERK_15 Activation
        v35=k33*ppMEK*ERK_15*UO
        #ERK_15 Deactivation
        v36=k34*ppERK_15
        #PEA binding to ERK
        v37=k37*ERK*PEA
        #PEA binding to ppERK
        v38=k38*ppERK*PEA
        #ERK_15 dissociation
        v39=k39*ERK_15
        #ppERK_15 dissociation
        v40=k40*ppERK_15
        #ppERK nuclear translocation
        v41=k41*ppERK
        #ppERKn deactivation and export
        v42=k42*ppERKn
        #PEA phosphorylation on 104
        v43=k43*ppERK*PEA*S104A*ERKDN
        #PEA phosphorylation on 116
        v44=k44*pAkt*PEA*S116A
        #PEA dephosphorylation on 104
        v45=k45*p104PEA
        #PEA dephosphorylation on 116
        v46=k46*p116PEA
        #Akt Activation
        v47=k47*PIP3*Akt
        #Akt Deactivation
        v48=k48*pAkt
        #Saturable EGFR endocytosis
        v49=k52*pEGFR/(k53+pEGFR)
        #pNGFR endocytosis
        v50=k54*pNGFR**k57/(k58**k57+pNGFR**k57)
        #Dephosphorylation of pEGFRi
        v51=k55*pEGFRi
        #Phosphorylation of EGFRi
        v52=k56*EGFRi
        #Phosphorylation of RSK
        v53=k68*(ppERK+ppERK_15)*RSK*ERKDN
        #Dephosphorylation of RSK
        v54=k69*pRSK
        # new Rap1 fluxes
        #Rap1 activation/deactivation
        v55=k_Rap1act*(pEGFR + pEGFRi)*Rap1 - k_Rap1deact*Rap1_act
        v56 = k_RafRap1*Raf*Rap1_act/(Kd_RafRap1 + Raf)


        # ODE rhs
        d_NGF = NGFDeriv
        trans_fun = lambda v1, v2: jnp.squeeze(-v1 + v2)
        sus_fun = lambda v1, v2: 0.0
        d_EGF = cond(self.transient, trans_fun, sus_fun, v1, v2)
        # if self.transient:
        # if self.transient:
        #     d_EGF = -v1 + v2
        # else:
        #     d_EGF = EGFDeriv
        d_NGFR = v2-v1
        d_pNGFR = -d_NGFR-v50
        d_EGFR = v4+v6-v3
        d_pEGFR = v3-v4-v5-v49
        d_pEGFRi = v5-v7+v49+v52-v51
        d_PIP3 = v8+v9-v10
        d_Akt = v48-v47
        d_pAkt = -d_Akt
        # d_PhE = 0.0
        d_PLCg = v13-v12-v11
        d_pPLCg = -d_PLCg
        d_PKC = v16-v14-v15
        d_PKCa = -d_PKC
        d_RKIP = v18-v17
        d_pRKIP = -d_RKIP
        d_RasD = v26-v19-v20-v21-v22-v23-v24-v25
        d_RasT = -d_RasD
        d_NF1 = v28-v27
        d_pNF1 = -d_NF1
        d_Raf = v30-v29 - v56 # v56 is new Rap1 dependent term
        d_Rafa = -d_Raf
        d_MEK = v32-v31
        d_ppMEK = -d_MEK
        d_ERK = v34-v33-v37+v39+v42*V_NUC/V_CYT
        d_ppERK = v33-v34-v38+v40-v41
        d_ERK_15 = v36-v35-v39+v37
        d_ppERK_15 = v35-v36-v40+v38
        d_ppERKn = v41*V_CYT/V_NUC-v42
        d_PEA = v39+v40-v37-v38-v43-v44+v45+v46
        d_p104PEA = v43-v45
        d_p116PEA = v44-v46
        d_EGFRi = -v52+v51-v6
        # d_NGFRi = 0.0
        d_RSK = v54-v53
        d_pRSK = -d_RSK
        d_Rap1 = -v55
        d_Rap1_act = v55

        # concatenate into tuple and return
        return (d_EGF, d_EGFR, d_pEGFR, d_pEGFRi, 
        d_PIP3, d_Akt, d_pAkt, d_PLCg, d_pPLCg, d_PKC, d_PKCa, 
        d_RKIP, d_pRKIP, d_RasD, d_RasT, d_NF1, d_pNF1, d_Raf, d_Rafa, 
        d_MEK, d_ppMEK, d_ERK, d_ppERK, d_ERK_15, d_ppERK_15, d_ppERKn, 
        d_PEA, d_p104PEA, d_p116PEA, d_EGFRi, d_RSK, d_pRSK, d_Rap1, d_Rap1_act)

        # commented on 3/19/24 to exclude NGF/NGFR from the model
        # return (d_NGF, d_EGF, d_NGFR, d_pNGFR, d_EGFR, d_pEGFR, d_pEGFRi, 
        # d_PIP3, d_Akt, d_pAkt, d_PhE, d_PLCg, d_pPLCg, d_PKC, d_PKCa, 
        # d_RKIP, d_pRKIP, d_RasD, d_RasT, d_NF1, d_pNF1, d_Raf, d_Rafa, 
        # d_MEK, d_ppMEK, d_ERK, d_ppERK, d_ERK_15, d_ppERK_15, d_ppERKn, 
        # d_PEA, d_p104PEA, d_p116PEA, d_EGFRi, d_NGFRi, d_RSK, d_pRSK)


    def get_nominal_params(self):

        p_dict =  {
            'k1': 1.95e-4,
            'k2': 8.81e-3,
            'k3': 0.0014,
            'k4': 0.0099,
            'k5': 0.0002,
            'k6': 0.3993,
            'k7': 0.0698,
            'k8':  0.633203292,
            'k9':  0.095749937,
            'k10': 0.232589605,
            'k11': 1.356614335,
            'k12': 0.096312665,
            'k13': 0.117816533,
            'k14': 9.8795,
            'k15': 1.7561,
            'k16': 0.0012,
            'k17': 4.8118,
            'k18': 0.5036,
            'k19': 1.3736, 
            'k20': 0.8445, 
            'k21': 0.3484, 
            'k22': 0.1000, 
            'k23':  0.0000, 
            'k24': 0.0371, 
            'k25': 0.0400, 
            'k26': 1.0000, 
            'k27': 0.1000, 
            'k28': 0.1000, 
            'k29': 0.1720, 
            'k30': 0.0100, 
            'k31': 1.4939, 
            'k32': 0.6585, 
            'k33': 0.1000, 
            'k34': 0.0100, 
            'k35': 1.3301, 
            'k36': 0.0513, 
            'k37': 1.0000,
            'k38': 0.9781,
            'k39': 0.0010,
            'k40': 1.2498,
            'k41': 0.0100,
            'k42': 0.1000,
            'k43': 0.1000,
            'k44': 5.0000,
            'k45': 1.0000,
            'k46': 1.0000,
            'k47': 0.790345221,
            'k48': 0.252620236,
            'k49': 0.0010, 
            'k50': 0.3662, 
            'k51': 2.7800, 
            'k52': 0.0013,
            'k53': 0.9979,
            'k54': 0.005,
            'k55': 0.0088,
            'k56': 0.0023,
            'k57': 2.0,
            'k58': 0.998,
            'k59': 0.6621,
            'k60': 0.1820,
            'k61': 9.7854,
            'k62': 0.9034,
            'k63': 0.8963,
            'k64': 7.4426,
            'k65': 0.3541,
            'k66': 2.3599,
            'k67': 3.8536,
            'k68': 0.0005, 
            'k69':  0.0000, 
            'k70': 3.0000, 
            'k71': 0.2896, 
            'k72': 3.0000, 
            'k73': 0.2000,
            'V_NUC':2.12e-13, #liters
            'V_CYT':9.38e-13, #liters
            'NGFDeriv': 0.0,
            'EGFDeriv': 0.0,
            # new Rap1 parameters
            'k_Rap1act': 0.5,
            'k_Rap1deact': 0.5,
            'k_RafRap1': 0.5,
            'Kd_RafRap1': 1.0,
        }

        p_list = [p_dict[key] for key in p_dict.keys()]

        return p_dict, p_list



    def get_initial_conditions(self):

        y0_dict = {
            # commented on 3/19/24 to exclude NGF/NGFR from the model
            # 'NGF':0.0,
            'EGF':0.0,
            # commented on 3/19/24 to exclude NGF/NGFR from the model
            # 'NGFR':0.0, # set to 1.0 to recover results in the paper
            # 'pNGFR':0.0,
            'EGFR':1.0,
            'pEGFR':0.0,
            'pEGFRi':0.0,
            'PIP3':0.0,
            'Akt':1.0,
            'pAkt':0.0,
            # 'PhE':0.0, # commented on 3/19/24 to fix issues
            'PLCg':1.0,
            'pPLCg':0.0,
            'PKC':1.0,
            'PKCa':0.0,
            'RKIP':1.0,
            'pRKIP':0.0,
            'RasD':1.0,
            'RasT':0.0,
            'NF1':1.0,
            'pNF1':0.0,
            'Raf':1.0,
            'Rafa':0.0,
            'MEK':1.0,
            'ppMEK':0.0,
            'ERK':0.0311267292018,
            'ppERK':0.0,
            'ERK_15':0.9688732707982,
            'ppERK_15':0.0,
            'ppERKn':0.0,
            'PEA':0.0311267292018,
            'p104PEA':0.0,
            'p116PEA':0.0,
            'EGFRi':0.0,
            # 'NGFRi':0.0, # commented on 3/19/24 to fix issues
            'RSK':1.0,
            'pRSK':0.0,
            'Rap1':1.0,
            'Rap1_act':0.0,
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup