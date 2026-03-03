import equinox as eqx
import jax.numpy as jnp
from jax.lax import cond

class birtwistle_2007(eqx.Module):
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=True): # defaults to transient stim
        self.transient = transient

    def __call__(self, t, y, args):
        # unpack state
        E, H, E1, E2, E3, E4, E_E1, H_E3, H_E4, E11, E12, E23, E34, E24, E44, \
            E11P, E12P, E23P, E34P, E24P, E44P, G, S, I, R, O, A, E11G, E11S, \
            E11R, E12G, E12S, E12R, E23G, E23S, E23I, E23R, E34G, E34S, E34I, \
            E34R, E24G, E24S, E24I, E24R, E44G, E44S, E44I, E44R, SigG, SigS, \
            SigI, SigR, SigA, SigSP, SigAP, SigG_O, SigG_A, SigSP_G, SigAP_S, \
            SigAP_I, SigAP_R, P3_A, P2, P3, Akt, Aktstar, RsD, RsT, SigRP, Raf, \
            Rafstar, MEK, MEKstar, ERK, ERKstar, OP, AP, A_SigG_O, SigA_G, \
            SigA_G_O, SigO, E13, E14, E13P, E14P, E13G, E13S, E13I, E13R, E14G, \
            E14S, E14I, E14R, fint, T, E11T, E12T, E23T, E34T, E24T, E44T, \
            E13T, E14T, SigAP_T, SigT, E1_PT, E2_PT, E4_PT, E_E1_PT, H_E4_PT, \
            pERK, ERK_MEKstar, pERK_MEKstar, ERKpase, ERKstar_ERKpase, \
            pERK_ERKpase = y

        # unpack parameters
        VmaxPY, KmPY, kdeg, \
        kf47, Vmaxr47, Kmf47, Kmr47, kf48, Kmf48, Kmr48, PTEN, kf49, kr49, \
        Kmf49, Kmr49, Kmr49b, kr49b, kf51, Vmaxr51, Kmf51, \
        Kmrb51, kf52, Vmaxr52, Kmf52, Kmr52, \
        kf54, Vmaxr54, Kmf54, Kmr54, kf55, Vmaxr55, Kmf55, Kmr55, kf38, kf39, \
        kf50, a98, b98, koff46, EGF_off, HRGoff_3, HRGoff_4, koff4, koff5, \
        koff6, koff7, koff8, koff9, koff57, koff16, koff17, koff18, koff19, \
        koff20, koff21, koff22, koff23, koff24, koff25, koff26, koff27, koff28, \
        koff29, koff30, koff31, koff32, koff33, koff34, koff35, koff36, koff37, \
        koff65, koff66, koff67, koff40, koff41, koff42, koff43, koff44, koff45, \
        koff58, koff59, koff68, kPTP10, kPTP11, kPTP12, kPTP13, \
        kPTP14, kPTP15, kPTP63, kPTP64, koff73, koff74, koff75, koff76, koff77, \
        koff78, koff79, koff80, kPTP38, kPTP39, koff88, kPTP50, kf81, Vmaxr81, \
        Kmf81, Kmr81, kf82, Vmaxr82, Kmf82, Kmr82, kf83, Vmaxr83, Kmf83, Kmr83, \
        kf84, Vmaxr84, Kmf84, Kmr84, kf85, Vmaxr85, Kmf85, Kmr85, kcon49, kon1, \
        kon86, kon2, kon3, kon87, kon4, kon5, kon6, kon7, kon8, kon9, kon57, \
        kf10, kf11, kf12, kf13, kf14, kf15, kf63, kf64, kon16, kon17, kon18, \
        kon73, kon19, kon20, kon21, kon74, kon22, kon23, kon24, kon25, kon75, \
        kon26, kon27, kon28, kon29, kon76, kon30, kon31, kon32, kon33, kon77, \
        kon34, kon35, kon36, kon37, kon78, kon79, kon65, kon66, kon67, kon80, \
        kon40, kon41, kon42, kon43, kon44, kon45, kon88, kon46, kon58, kon59, \
        kon60, VeVc, koff60, koff61, kon61, kon62, koff62, kon68, kon69, koff69, \
        kon70, koff70, kon71, koff71, kon72, koff72, eps, kon89, koff89, kcat90, \
        kon91, koff91, kcat92, kon93, koff93, kcat94, kon95, koff95, kcat96 = args
        
        # define fluxes
        J0 = ((2.0 * kon68 * E13P * R) - (koff68 * (SigR / (SigR + SigRP + eps)) * E13R))
        J1 = (((kf10 * E11) - ((VmaxPY * E11P) / (KmPY + E11P))) - (kPTP10 * SigT * E11P))
        J2 = (((kf55 * A * ERKstar) / (Kmf55 + A)) - ((Vmaxr55 * AP) / (Kmr55 + AP)))
        J3 = (((kf84 * E_E1 * ERKstar) / (Kmf84 + E_E1)) - ((Vmaxr84 * E_E1_PT) / (Kmr84 + E_E1_PT)))
        J4 = ((4.0 * kon31 * E24P * S) - (koff31 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E24S))
        J5 = (((kf85 * H_E4 * ERKstar) / (Kmf85 + H_E4)) - ((Vmaxr85 * H_E4_PT) / (Kmr85 + H_E4_PT)))
        J6 = ((4.0 * kon26 * E34P * G) - (koff26 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E34G))
        J7 = ((kon89 * ERK * MEKstar) - (koff89 * ERK_MEKstar))
        J8 = ((2.0 * kon72 * E14P * R) - (koff72 * (SigR / (SigR + SigRP + eps)) * E14R))
        J9 = (kdeg * E11R)
        J10 = ((kon8 * H_E4 * E2) - (koff8 * E24))
        J11 = ((2.0 * kon36 * E44P * I) - (koff36 * E44I))
        J12 = ((3.0 * kon23 * E23P * S) - (koff23 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E23S))
        J13 = ((2.0 * kon37 * E44P * R) - (koff37 * (SigR / (SigR + SigRP + eps)) * E44R))
        J14 = ((1.0 * kon71 * E14P * I) - (koff71 * E14I))
        J15 = ((kon40 * SigG * O) - (koff40 * SigG_O))
        J16 = ((kon46 * P3 * A) - (koff46 * P3_A * (SigA / (eps + SigA + SigAP + SigAP_S + SigAP_R + SigAP_I + SigAP_T))))
        J17 = ((2.0 * kon25 * E23P * R) - (koff25 * (SigR / (SigR + SigRP + eps)) * E23R))
        J18 = ((kon87 * H * E4_PT) - (HRGoff_4 * H_E4_PT))
        J19 = (((((((kon87 * H * E4_PT) - (HRGoff_4 * H_E4_PT)) + (kon2 * H * E3)) - (HRGoff_3 * H_E3)) + (kon3 * H * E4)) - (HRGoff_4 * H_E4)) / VeVc)
        J20 = ((kon9 * H_E4 * H_E4) - (koff9 * E44))
        J21 = ((1.0 * kon32 * E24P * I) - (koff32 * E24I))
        J22 = (((kf47 * P3 * Akt) / (Kmf47 + Akt)) - ((Vmaxr47 * Aktstar) / (Kmr47 + Aktstar)))
        J23 = ((kon91 * pERK * MEKstar) - (koff91 * pERK_MEKstar))
        J24 = (kcat90 * ERK_MEKstar)
        J25 = ((kon3 * H * E4) - (HRGoff_4 * H_E4))
        J26 = ((3.0 * kon44 * SigAP * I) - (koff44 * SigAP_I))
        J27 = ((kon42 * SigSP * G) - (koff42 * SigSP_G * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps))))
        J28 = ((kon58 * SigA_G * O) - (koff58 * SigA_G_O))
        J29 = ((6.0 * kon20 * E12P * S) - (koff20 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E12S))
        J30 = (a98 * ( - fint + b98))
        J31 = (((kf63 * E13) - ((VmaxPY * E13P) / (KmPY + E13P))) - (kPTP63 * SigT * E13P))
        J32 = (((((kon1 * E * E1) - (EGF_off * E_E1)) + (kon86 * E * E1_PT)) - (EGF_off * E_E1_PT)) / VeVc)
        J33 = ((2.0 * kon29 * E34P * R) - (koff29 * (SigR / (SigR + SigRP + eps)) * E34R))
        J34 = (kdeg * E11P)
        J35 = ((5.0 * kon66 * E13P * S) - (koff66 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E13S))
        J36 = ((4.0 * kon16 * E11P * G) - (koff16 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E11G))
        J37 = ((4.0 * kon73 * E11P * T) - (koff73 * E11T))
        J38 = ((2.0 * kon88 * SigAP * T) - (koff88 * SigAP_T))
        J39 = ((2.0 * kon21 * E12P * R) - (koff21 * (SigR / (SigR + SigRP + eps)) * E12R))
        J40 = ((3.0 * kon19 * E12P * G) - (koff19 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E12G))
        J41 = (((kf82 * E2 * ERKstar) / (Kmf82 + E2)) - ((Vmaxr82 * E2_PT) / (Kmr82 + E2_PT)))
        J42 = ((kon7 * H_E3 * H_E4) - (koff7 * E34))
        J43 = ((3.0 * kon74 * E12P * T) - (koff74 * E12T))
        J44 = (((kf64 * E14) - ((VmaxPY * E14P) / (KmPY + E14P))) - (kPTP64 * SigT * E14P))
        J45 = ((4.0 * kon69 * E14P * G) - (koff69 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E14G))
        J46 = (((kf83 * E4 * ERKstar) / (Kmf83 + E4)) - ((Vmaxr83 * E4_PT) / (Kmr83 + E4_PT)))
        J47 = ((3.0 * kon30 * E24P * G) - (koff30 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E24G))
        J48 = ((3.0 * kon43 * SigAP * S) - (koff43 * SigAP_S * (SigS / (SigS + SigSP + SigSP_G + eps))))
        J49 = ((2.0 * kon77 * E24P * T) - (koff77 * E24T))
        J50 = ((3.0 * kon22 * E23P * G) - (koff22 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E23G))
        J51 = (kcat92 * pERK_MEKstar)
        J52 = ((3.0 * kon79 * E13P * T) - (koff79 * E13T))
        J53 = ((kon5 * E_E1 * E2) - (koff5 * E12))
        J54 = ((kon59 * SigG_O * A) - (koff59 * A_SigG_O * (SigA / (eps + SigA + SigAP + SigAP_S + SigAP_R + SigAP_I + SigAP_T))))
        J55 = ((3.0 * kon27 * E34P * S) - (koff27 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E34S))
        J56 = ((kon2 * H * E3) - (HRGoff_3 * H_E3))
        J57 = ((4.0 * kon28 * E34P * I) - (koff28 * E34I))
        J58 = ((kon62 * H_E4 * E_E1) - (koff62 * E14))
        J59 = ((kon93 * ERKstar * ERKpase) - (koff93 * ERKstar_ERKpase))
        J60 = ((2.0 * kon75 * E23P * T) - (koff75 * E23T))
        J61 = (kcat94 * ERKstar_ERKpase)
        J62 = (kdeg * E11T)
        J63 = (((kf11 * E12) - ((VmaxPY * E12P) / (KmPY + E12P))) - (kPTP11 * SigT * E12P))
        J64 = ((kon86 * E * E1_PT) - (EGF_off * E_E1_PT))
        J65 = ((2.0 * kon18 * E11P * R) - (koff18 * (SigR / (SigR + SigRP + eps)) * E11R))
        J66 = ((3.0 * kon24 * E23P * I) - (koff24 * E23I))
        J67 = (kdeg * E11S)
        J68 = (((((kf49 * SigO * RsD) / (Kmf49 + RsD)) - ((kr49 * SigR * RsT) / (Kmr49 + RsT))) - ((kr49b * SigRP * RsT) / (Kmr49b + RsT))) - (kcon49 * RsT))
        J69 = ((3.0 * kon80 * E14P * T) - (koff80 * E14T))
        J70 = (((kf14 * E24) - ((VmaxPY * E24P) / (KmPY + E24P))) - (kPTP14 * SigT * E24P))
        J71 = ((kon1 * E * E1) - (EGF_off * E_E1)) / VeVc
        J72 = ((8.0 * kon17 * E11P * S) - (koff17 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E11S))
        J73 = ((kon57 * P3_A * G) - (koff57 * SigA_G))
        J74 = (((kf54 * O * ERKstar) / (Kmf54 + O)) - ((Vmaxr54 * OP) / (Kmr54 + OP)))
        J75 = ((4.0 * kon35 * E44P * S) - (koff35 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E44S))
        J76 = (((kf38 * SigS * (E11P + E12P + E23P + E24P + E34P + E44P + E13P + E14P)) - ((VmaxPY * SigSP) / (KmPY + SigSP))) - (kPTP38 * SigT * SigSP))
        J77 = (kcat96 * pERK_ERKpase)
        J78 = (kdeg * E11G)
        J79 = (((kf50 * SigR * (E11P + E12P + E23P + E24P + E34P + E44P + E13P + E14P)) - ((VmaxPY * SigRP) / (KmPY + SigRP))) - (kPTP50 * SigT * SigRP))
        J80 = (((kf12 * E23) - ((VmaxPY * E23P) / (KmPY + E23P))) - (kPTP12 * SigT * E23P))
        J81 = (((kf13 * E34) - ((VmaxPY * E34P) / (KmPY + E34P))) - (kPTP13 * SigT * E34P))
        J82 = (((kf15 * E44) - ((VmaxPY * E44P) / (KmPY + E44P))) - (kPTP15 * SigT * E44P))
        J83 = ((kon41 * SigG * A) - (koff41 * SigG_A * (SigA / (eps + SigA + SigAP + SigAP_S + SigAP_R + SigAP_I + SigAP_T))))
        J84 = ((kon61 * H_E3 * E_E1) - (koff61 * E13))
        J85 = ((kon60 * SigG_A * O) - (koff60 * A_SigG_O))
        J86 = ((6.0 * kon70 * E14P * S) - (koff70 * (SigS / (SigS + SigSP + SigSP_G + eps)) * E14S))
        J87 = ((2.0 * kon33 * E24P * R) - (koff33 * (SigR / (SigR + SigRP + eps)) * E24R))
        J88 = ((4.0 * kon34 * E44P * G) - (koff34 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E44G))
        J89 = (((kf81 * E1 * ERKstar) / (Kmf81 + E1)) - ((Vmaxr81 * E1_PT) / (Kmr81 + E1_PT)))
        J90 = ((kon95 * pERK * ERKpase) - (koff95 * pERK_ERKpase))
        J91 = ((3.0 * kon67 * E13P * I) - (koff67 * E13I))
        J92 = (((kf48 * (1.0 - (fint * (E11P / (E11P + E12P + E23P + E24P + E34P + E44P + E13P + E14P + eps)))) * SigI * P2) / (Kmf48 + P2)) - ((3.0 * PTEN * P3) / (Kmr48 + P3)))
        J93 = ((kon6 * H_E3 * E2) - (koff6 * E23))
        J94 = (((kf51 * RsT * Raf) / (Kmf51 + Raf)) - ((Vmaxr51 * Rafstar) / (Kmrb51 + Rafstar)))
        J95 = ((2.0 * kon45 * SigAP * R) - (koff45 * SigAP_R * (SigR / (SigR + SigRP + eps))))
        J96 = ((4.0 * kon65 * E13P * G) - (koff65 * (SigG / (SigG + SigG_A + SigG_O + A_SigG_O + eps)) * E13G))
        J97 = (((kf52 * Rafstar * MEK) / (Kmf52 + MEK)) - ((Vmaxr52 * MEKstar) / (Kmr52 + MEKstar)))
        J98 = ((2.0 * kon76 * E34P * T) - (koff76 * E34T))
        J99 = ((kon4 * E_E1 * E_E1) - (koff4 * E11))
        J100 = (((kf39 * SigA * (E11P + E12P + E23P + E24P + E34P + E44P + E13P + E14P)) - ((VmaxPY * SigAP) / (KmPY + SigAP))) - (kPTP39 * SigT * SigAP))
        J101 = ((2.0 * kon78 * E44P * T) - (koff78 * E44T))
        
        # variable EGF rate
        trans_fun = lambda J32: jnp.squeeze(-J32)
        sus_fun = lambda J32: 0.0
        d_E = cond(self.transient, trans_fun, sus_fun, J32)
        # if self.transient:
        #     d_E = -J32
        # else:
        #     d_E = 0.0
        d_H = -J19
        d_E1 = -J71 -J89
        d_E2 = -J53 -J93 -J10 -J41
        d_E3 = -J56
        d_E4 = -J25 -J46
        d_E_E1 = ((J71 ) - (2.0 * J99 ) - (J53 ) - (J84 ) - (J58 ) - (J3 ))
        d_H_E3 = ((J56 ) - (J93 ) - (J42 ) - (J84 ))
        d_H_E4 = ((J25 ) - (J42 ) - (J10 ) - (2.0 * J20 ) - (J58 ) - (J5 ))
        d_E11 = ((J99 ) - (J1 ))
        d_E12 = ((J53 ) - (J63 ))
        d_E23 = ((J93 ) - (J80 ))
        d_E34 = ((J42 ) - (J81 ))    
        d_E24 = ((J10 ) - (J70 ))    
        d_E44 = ((J20 ) - (J82 ))    
        d_E11P = ((J1 ) - (J36 ) - (J72 ) - (J65 ) - (J37 ) - (J34 ))    
        d_E12P = ((J63 ) - (J40 ) - (J29 ) - (J39 ) - (J43 ))    
        d_E23P = ((J80 ) - (J50 ) - (J12 ) - (J66 ) - (J17 ) - (J60 ))    
        d_E34P = ((J81 ) - (J6 ) - (J55 ) - (J57 ) - (J33 ) - (J98 ))    
        d_E24P = ((J70 ) - (J47 ) - (J4 ) - (J21 ) - (J87 ) - (J49 ))    
        d_E44P = ((J82 ) - (J88 ) - (J75 ) - (J11 ) - (J13 ) - (J101 ))    
        d_G = ( - (J36 ) - (J40 ) - (J50 ) - (J6 ) - (J47 ) - (J88 ) - (J27 ) - (J73 ) - (J96 ) - (J45 ) + (J78 ))    
        d_S = ( - (J72 ) - (J29 ) - (J12 ) - (J55 ) - (J4 ) - (J75 ) - (J48 ) - (J35 ) - (J86 ) + (J67 ))    
        d_I = ( - (J66 ) - (J57 ) - (J21 ) - (J11 ) - (J26 ) - (J91 ) - (J14 ))    
        d_R = ( - (J65 ) - (J39 ) - (J17 ) - (J33 ) - (J87 ) - (J13 ) - (J95 ) - (J0 ) - (J8 ) + (J9 ))    
        d_O = ( - (J15 ) - (J74 ) - (J28 ) - (J85 ))    
        d_A = ( - (J83 ) - (J16 ) - (J2 ) - (J54 ))    
        d_E11G = ((J36 ) - (J78 ))    
        d_E11S = ((J72 ) - (J67 ))    
        d_E11R = ((J65 ) - (J9 ))    
        d_E12G = (J40 )    
        d_E12S = (J29 )    
        d_E12R = (J39 )    
        d_E23G = (J50 )    
        d_E23S = (J12 )    
        d_E23I = (J66 )    
        d_E23R = (J17 )    
        d_E34G = (J6 )    
        d_E34S = (J55 )    
        d_E34I = (J57 )    
        d_E34R = (J33 )    
        d_E24G = (J47 )    
        d_E24S = (J4 )    
        d_E24I = (J21 )    
        d_E24R = (J87 )    
        d_E44G = (J88 )    
        d_E44S = (J75 )    
        d_E44I = (J11 )    
        d_E44R = (J13 )    
        d_SigG = ((J36 ) + (J40 ) + (J50 ) + (J6 ) + (J47 ) + (J88 ) - (J15 ) - (J83 ) + (J27 ) + (J96 ) + (J45 ) - (J78 ))    
        d_SigS = ((J72 ) + (J29 ) + (J12 ) + (J55 ) + (J4 ) + (J75 ) - (J76 ) + (J48 ) + (J35 ) + (J86 ) - (J67 ))    
        d_SigI = ((J66 ) + (J57 ) + (J21 ) + (J11 ) + (J26 ) + (J91 ) + (J14 ))    
        d_SigR = ((J65 ) + (J39 ) + (J17 ) + (J33 ) + (J87 ) + (J13 ) + (J95 ) - (J79 ) + (J0 ) + (J8 ) - (J9 ))    
        d_SigA = ( - (J100 ) + (J83 ) + (J16 ) + (J54 ))    
        d_SigSP = ((J76 ) - (J27 ))    
        d_SigAP = ((J100 ) - (J48 ) - (J26 ) - (J95 ) - (J38 ))    
        d_SigG_O = ((J15 ) - (J54 ))    
        d_SigG_A = ((J83 ) - (J85 ))    
        d_SigSP_G = (J27 )    
        d_SigAP_S = (J48 )    
        d_SigAP_I = (J26 )    
        d_SigAP_R = (J95 )    
        d_P3_A = ((J16 ) - (J73 ))    
        d_P2 = - (J92 )    
        d_P3 = ( - (J16 ) + (J92 ))    
        d_Akt = - (J22 )    
        d_Aktstar = (J22 )    
        d_RsD = - (J68 )    
        d_RsT = (J68 )    
        d_SigRP = (J79 )    
        d_Raf = - (J94 )    
        d_Rafstar = (J94 )    
        d_MEK = - (J97 )    
        d_MEKstar = ((J97 ) - (J7 ) + (J24 ) - (J23 ) + (J51 ))    
        d_ERK = ( - (J7 ) + (J77 ))    
        d_ERKstar = ((J51 ) - (J59 ))    
        d_OP = (J74 )    
        d_AP = (J2 )    
        d_A_SigG_O = ((J54 ) + (J85 ))    
        d_SigA_G = ((J73 ) - (J28 ))    
        d_SigA_G_O = (J28 )    
        d_SigO = ((J15 ) + (J28 ) + (J85 ))    
        d_E13 = ((J84 ) - (J31 ))    
        d_E14 = ((J58 ) - (J44 ))    
        d_E13P = ((J31 ) - (J96 ) - (J35 ) - (J91 ) - (J0 ) - (J52 ))    
        d_E14P = ((J44 ) - (J45 ) - (J86 ) - (J14 ) - (J8 ) - (J69 ))    
        d_E13G = (J96 )    
        d_E13S = (J35 )    
        d_E13I = (J91 )    
        d_E13R = (J0 )    
        d_E14G = (J45 )    
        d_E14S = (J86 )    
        d_E14I = (J14 )    
        d_E14R = (J8 )    
        d_fint = (J30 )    
        d_T = ( - (J37 ) - (J43 ) - (J60 ) - (J98 ) - (J49 ) - (J101 ) - (J52 ) - (J69 ) - (J38 ) + (J62 ))    
        d_E11T = ((J37 ) - (J62 ))    
        d_E12T = (J43 )    
        d_E23T = (J60 )    
        d_E34T = (J98 )    
        d_E24T = (J49 )    
        d_E44T = (J101 )    
        d_E13T = (J52 )    
        d_E14T = (J69 )    
        d_SigAP_T = (J38 )    
        d_SigT = ((J37 ) + (J43 ) + (J60 ) + (J98 ) + (J49 ) + (J101 ) + (J52 ) + (J69 ) + (J38 ) - (J62 ))    
        d_E1_PT = ((J89 ) - (J64 ))    
        d_E2_PT = (J41 )    
        d_E4_PT = ((J46 ) - (J18 ))    
        d_E_E1_PT = ((J3 ) + (J64 ))    
        d_H_E4_PT = ((J5 ) + (J18 ))    
        d_pERK = ((J24 ) - (J23 ) + (J61 ) - (J90 ))    
        d_ERK_MEKstar = ((J7 ) - (J24 ))    
        d_pERK_MEKstar = ((J23 ) - (J51 ))    
        d_ERKpase = ( - (J59 ) + (J61 ) - (J90 ) + (J77 ))    
        d_ERKstar_ERKpase = ((J59 ) - (J61 ))    
        d_pERK_ERKpase = ((J90 ) - (J77 )) 

        return (
            d_E, d_H, d_E1, d_E2, d_E3, d_E4, d_E_E1, d_H_E3, d_H_E4, d_E11, d_E12, 
            d_E23, d_E34, d_E24, d_E44, d_E11P, d_E12P, d_E23P, d_E34P, 
            d_E24P, d_E44P, d_G, d_S, d_I, d_R, d_O, d_A, d_E11G, d_E11S, 
            d_E11R, d_E12G, d_E12S, d_E12R, d_E23G, d_E23S, d_E23I, d_E23R, 
            d_E34G, d_E34S, d_E34I, d_E34R, d_E24G, d_E24S, d_E24I, d_E24R, 
            d_E44G, d_E44S, d_E44I, d_E44R, d_SigG, d_SigS, d_SigI, d_SigR, 
            d_SigA, d_SigSP, d_SigAP, d_SigG_O, d_SigG_A, d_SigSP_G, d_SigAP_S, 
            d_SigAP_I, d_SigAP_R, d_P3_A, d_P2, d_P3, d_Akt, d_Aktstar, d_RsD, 
            d_RsT, d_SigRP, d_Raf, d_Rafstar, d_MEK, d_MEKstar, d_ERK, 
            d_ERKstar, d_OP, d_AP, d_A_SigG_O, d_SigA_G, d_SigA_G_O, d_SigO, 
            d_E13, d_E14, d_E13P, d_E14P, d_E13G, d_E13S, d_E13I, d_E13R, 
            d_E14G, d_E14S, d_E14I, d_E14R, d_fint, d_T, d_E11T, d_E12T, d_E23T, 
            d_E34T, d_E24T, d_E44T, d_E13T, d_E14T, d_SigAP_T, d_SigT, d_E1_PT, 
            d_E2_PT, d_E4_PT, d_E_E1_PT, d_H_E4_PT, d_pERK, d_ERK_MEKstar, d_pERK_MEKstar, 
            d_ERKpase, d_ERKstar_ERKpase, d_pERK_ERKpase)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        ic_dict = {
            'E': 0.0,
            'H': 0.0,
            'E1': 274.0,
            'E2': 158.0,
            'E3': 0.0, #294.0, set back to these to recover og results
            'E4': 0.0, #399.0,
            'E_E1': 0.0,
            'H_E3': 0.0,
            'H_E4': 0.0,
            'E11': 0.0,
            'E12': 0.0,
            'E23': 0.0,
            'E34': 0.0,
            'E24': 0.0,
            'E44': 0.0,
            'E11P': 0.0,
            'E12P': 0.0,
            'E23P': 0.0,
            'E34P': 0.0,
            'E24P': 0.0,
            'E44P': 0.0,
            'G': 82.4,
            'S': 11.5,
            'I': 46.4,
            'R': 93.6,
            'O': 82.3,
            'A': 43.1,
            'E11G': 0.0,
            'E11S': 0.0,
            'E11R': 0.0,
            'E12G': 0.0,
            'E12S': 0.0,
            'E12R': 0.0,
            'E23G': 0.0,
            'E23S': 0.0,
            'E23I': 0.0,
            'E23R': 0.0,
            'E34G': 0.0,
            'E34S': 0.0,
            'E34I': 0.0,
            'E34R': 0.0,
            'E24G': 0.0,
            'E24S': 0.0,
            'E24I': 0.0,
            'E24R': 0.0,
            'E44G': 0.0,
            'E44S': 0.0,
            'E44I': 0.0,
            'E44R': 0.0,
            'SigG': 0.0,
            'SigS': 0.0,
            'SigI': 0.0,
            'SigR': 0.0,
            'SigA': 0.0,
            'SigSP': 0.0,
            'SigAP': 0.0,
            'SigG_O': 0.0,
            'SigG_A': 0.0,
            'SigSP_G': 0.0,
            'SigAP_S': 0.0,
            'SigAP_I': 0.0,
            'SigAP_R': 0.0,
            'P3_A': 0.0,
            'P2': 197.0,
            'P3': 0.0,
            'Akt': 444.0,
            'Aktstar': 0.0,
            'RsD': 95.7,
            'RsT': 0.0,
            'SigRP': 0.0,
            'Raf': 743.0,
            'Rafstar': 0.0,
            'MEK': 772.0,
            'MEKstar': 0.0,
            'ERK': 750.0,
            'ERKstar': 0.0,
            'OP': 0.0,
            'AP': 0.0,
            'A_SigG_O': 0.0,
            'SigA_G': 0.0,
            'SigA_G_O': 0.0,
            'SigO': 0.0,
            'E13': 0.0,
            'E14': 0.0,
            'E13P': 0.0,
            'E14P': 0.0,
            'E13G': 0.0,
            'E13S': 0.0,
            'E13I': 0.0,
            'E13R': 0.0,
            'E14G': 0.0,
            'E14S': 0.0,
            'E14I': 0.0,
            'E14R': 0.0,
            'fint': 0.0,
            'T': 500.0,
            'E11T': 0.0,
            'E12T': 0.0,
            'E23T': 0.0,
            'E34T': 0.0,
            'E24T': 0.0,
            'E44T': 0.0,
            'E13T': 0.0,
            'E14T': 0.0,
            'SigAP_T': 0.0,
            'SigT': 0.0,
            'E1_PT': 0.0,
            'E2_PT': 0.0,
            'E4_PT': 0.0,
            'E_E1_PT': 0.0,
            'H_E4_PT': 0.0,
            'pERK': 0.0,
            'ERK_MEKstar': 0.0,
            'pERK_MEKstar': 0.0,
            'ERKpase': 0.0,
            'ERKstar_ERKpase': 35.0,
            'pERK_ERKpase': 0.0,}
        
        ic_tup = tuple([ic_dict[key] for key in ic_dict])

        return ic_dict, ic_tup
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        param_dict = {
            'VmaxPY': 223.8776,
            'KmPY': 486.1398,
            'kdeg': 0.0259,
            'kf47': 24.6048,
            'Vmaxr47': 590.5058,
            'Kmf47': 698.6022,
            'Kmr47': 483.8622,
            'kf48': 16.833,
            'Kmf48': 715.5688,
            'Kmr48': 324.9294,
            'PTEN': 693.5786,
            'kf49': 44.3501,
            'kr49': 552.6746,
            'Kmf49': 343.2483,
            'Kmr49': 753.1667,
            'Kmr49b': 381.2208,
            'kr49b': 640.8212,
            'kf51': 3.6515,
            'Vmaxr51': 16.737,
            'Kmf51': 599.7076,
            'Kmrb51': 988.4496,
            'kf52': 0.7745,
            'Vmaxr52': 199.2773,
            'Kmf52': 545.4408,
            'Kmr52': 675.2994,
            'kf54': 0.0538,
            'Vmaxr54': 588.2671,
            'Kmf54': 457.9645,
            'Kmr54': 336.183,
            'kf55': 0.2256,
            'Vmaxr55': 646.9003,
            'Kmf55': 460.9446,
            'Kmr55': 643.9247,
            'kf38': 279.9929,
            'kf39': 385.7428,
            'kf50': 389.1061,
            'a98': 0.0849,
            'b98': 0.1833,
            'koff46': 0.5194,
            'EGF_off': 0.0175,
            'HRGoff_3': 0.0009,
            'HRGoff_4': 0.0973,
            'koff4': 0.1717,
            'koff5': 4.3985,
            'koff6': 2.6619,
            'koff7': 8.0557,
            'koff8': 9.1034,
            'koff9': 5.5425,
            'koff57': 0.4526,
            'koff16': 0.5737,
            'koff17': 4.6874,
            'koff18': 2.2768,
            'koff19': 2.3361,
            'koff20': 0.6761,
            'koff21': 4.7291,
            'koff22': 3.6962,
            'koff23': 2.3619,
            'koff24': 4.4226,
            'koff25': 2.225,
            'koff26': 0.0103,
            'koff27': 1.8922,
            'koff28': 4.6432,
            'koff29': 2.0148,
            'koff30': 4.9936,
            'koff31': 1.2204,
            'koff32': 3.8752,
            'koff33': 1.2817,
            'koff34': 3.2036,
            'koff35': 1.8696,
            'koff36': 1.2567,
            'koff37': 0.4059,
            'koff65': 0.1185,
            'koff66': 2.2988,
            'koff67': 1.6142,
            'koff40': 3.1051,
            'koff41': 7.0487,
            'koff42': 3.5195,
            'koff43': 0.5441,
            'koff44': 0.4265,
            'koff45': 3.9967,
            'koff58': 6.3059,
            'koff59': 9.172,
            'koff68': 2.8871,
            'kPTP10': 29.8531,
            'kPTP11': 78.204,
            'kPTP12': 11.4211,
            'kPTP13': 55.2104,
            'kPTP14': 57.7506,
            'kPTP15': 60.2628,
            'kPTP63': 7.4766,
            'kPTP64': 48.6335,
            'koff73': 3.0048,
            'koff74': 1.2496,
            'koff75': 1.4323,
            'koff76': 2.1542,
            'koff77': 1.2237,
            'koff78': 0.2007,
            'koff79': 1.1852,
            'koff80': 2.9373,
            'kPTP38': 83.4465,
            'kPTP39': 79.6132,
            'koff88': 3.9255,
            'kPTP50': 96.5716,
            'kf81': 1.361,
            'Vmaxr81': 242.6034,
            'Kmf81': 485.2626,
            'Kmr81': 323.4012,
            'kf82': 6.9987,
            'Vmaxr82': 398.1931,
            'Kmf82': 781.4374,
            'Kmr82': 595.8395,
            'kf83': 1.7628,
            'Vmaxr83': 534.0531,
            'Kmf83': 609.4766,
            'Kmr83': 653.5184,
            'kf84': 4.6894,
            'Vmaxr84': 634.1626,
            'Kmf84': 622.3847,
            'Kmr84': 258.4637,
            'kf85': 6.7591,
            'Vmaxr85': 369.2261,
            'Kmf85': 179.6486,
            'Kmr85': 290.7667,
            'kcon49': 9.9783,
            'kon1': 0.00010086,
            'kon86': 0.0038,
            'kon2': 0.0059,
            'kon3': 0.0334,
            'kon87': 0.0008,
            'kon4': 0.5005,
            'kon5': 2.5427,
            'kon6': 0.2283,
            'kon7': 1.0606,
            'kon8': 1.0259,
            'kon9': 2.2868,
            'kon57': 0.0039,
            'kf10': 0.6496,
            'kf11': 0.3721,
            'kf12': 1.8012,
            'kf13': 0.8875,
            'kf14': 6.1726,
            'kf15': 1.3565,
            'kf63': 0.9297,
            'kf64': 1.2083,
            'kon16': 0.0097,
            'kon17': 0.0049,
            'kon18': 0.0117,
            'kon73': 0.0116,
            'kon19': 0.0896,
            'kon20': 0.0478,
            'kon21': 0.0114,
            'kon74': 0.0133,
            'kon22': 0.0007,
            'kon23': 0.0138,
            'kon24': 0.005,
            'kon25': 0.0995,
            'kon75': 0.0137,
            'kon26': 0.0355,
            'kon27': 0.0201,
            'kon28': 0.0074,
            'kon29': 0.0346,
            'kon76': 0.0053,
            'kon30': 0.002,
            'kon31': 0.0032,
            'kon32': 0.0009,
            'kon33': 0.0335,
            'kon77': 0.0101,
            'kon34': 0.0001,
            'kon35': 0.0602,
            'kon36': 0.0043,
            'kon37': 0.0791,
            'kon78': 0.0076,
            'kon79': 0.0078,
            'kon65': 0.0123,
            'kon66': 0.00019264,
            'kon67': 6.6667e-05,
            'kon80': 0.0002,
            'kon40': 0.0191,
            'kon41': 0.0051,
            'kon42': 0.0023,
            'kon43': 0.0127,
            'kon44': 0.0122,
            'kon45': 0.0028,
            'kon88': 0.0108,
            'kon46': 0.0148,
            'kon58': 0.0215,
            'kon59': 0.0077,
            'kon60': 0.00011994,
            'VeVc': 33.3,
            'koff60': 4.9981,
            'koff61': 5.229,
            'kon61': 0.8048,
            'kon62': 1.782,
            'koff62': 5.5142,
            'kon68': 0.0045,
            'kon69': 0.0084,
            'koff69': 3.97,
            'kon70': 0.0116,
            'koff70': 2.6069,
            'kon71': 0.0078,
            'koff71': 2.2988,
            'kon72': 0.0355,
            'koff72': 0.907,
            'eps': 1e-16,
            'kon89': 0.1997,
            'koff89': 99.9637,
            'kcat90': 20.0037,
            'kon91': 0.1966,
            'koff91': 99.9983,
            'kcat92': 0.2004,
            'kon93': 0.2003,
            'koff93': 100.0037,
            'kcat94': 0.9966,
            'kon95': 0.1993,
            'koff95': 100.0023,
            'kcat96': 19.9851,}


        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

