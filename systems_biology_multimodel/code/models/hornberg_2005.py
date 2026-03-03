import equinox as eqx
import jax.numpy as jnp
from jax.lax import cond

class hornberg_2005(eqx.Module):
    """ Note this is the code at https://models.physiomeproject.org/exposure/48c4b41256d698c4d18aace1cb159865/hornberg_binder_bruggeman_schoeberl_heinrich_westerhoff_2005.cellml/@@cellml_codegen/Python
    
    Which comes from the Hornberg paper entry on the CellML site https://models.physiomeproject.org/exposure/48c4b41256d698c4d18aace1cb159865

    c1 - EGF
    c2 - EGFR
    c3 - EGFR_EGF
    """
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=False): # defaults to sustained stim
        self.transient = transient

    def __call__(self, t, y, constants):
        # unpack state
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, \
            c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, \
            c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, \
            c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, \
            c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, \
            c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, \
            c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, \
            c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103 = y

        v1 = constants[0]*c1*c2-constants[1]*c3
        # variable EGF rate
        trans_fun = lambda v1: jnp.squeeze(-v1)
        sus_fun = lambda v1: 0.0
        dy_dt_0 = cond(self.transient, trans_fun, sus_fun, v1)
        # if self.transient:
        #     dy_dt_0 = -v1
        # else:
        #     dy_dt_0 = constants[95]

        v2 = constants[2]*c3*c3-constants[3]*c4
        dy_dt_2 = v1-2.0*v2
        v3 = constants[4]*c4*1.0-constants[5]*c5
        dy_dt_3 = v2-v3
        v4 = constants[6]*c23*c12-constants[7]*c7
        v5 = constants[8]*c18*c9-constants[9]*c7
        dy_dt_6 = v4+v5
        v6 = constants[10]*c2-constants[11]*c6
        v13 = constants[16]*1.0-constants[17]*c2
        dy_dt_1 = v13-(v1+v6)
        v7 = constants[10]*c5-constants[11]*c8
        v8 = constants[12]*c5*c14-constants[13]*c15
        dy_dt_4 = v3-(v7+v8)
        v10 = constants[14]*c6*c16-constants[15]*c10
        v60 = constants[87]*c6-constants[88]*c86
        dy_dt_5 = v6-(v10+v60)
        v11 = constants[2]*c10*c10-constants[3]*c11
        dy_dt_9 = v10-2.0*v11
        v61 = constants[89]*c16-constants[90]*c13
        dy_dt_12 = v61
        dy_dt_15 = -(v10+v61)
        dy_dt_85 = v60
        v12 = constants[4]*c11-constants[5]*c8
        dy_dt_10 = v11-v12
        v14 = constants[12]*c8*c14-constants[13]*c17
        dy_dt_13 = -(v8+v14)
        v16 = constants[20]*c22*c15-constants[21]*c23
        v9 = constants[10]*c23-constants[11]*c18
        v17 = constants[22]*c24*c23-constants[23]*c25
        dy_dt_22 = v16-(v4+v9+v17)
        v19 = constants[26]*c28*c25-constants[27]*c27
        v27 = constants[26]*c35*c28-constants[27]*c36
        v28 = constants[39]*c28*c41-constants[40]*c42
        dy_dt_27 = -(v19+v27+v28)
        v29 = constants[41]*c43*c45-constants[42]*c42
        dy_dt_41 = v28+v29
        v20 = constants[28]*c25*c43-constants[29]*c29
        v30 = constants[28]*c35*c43-constants[29]*c37
        dy_dt_42 = -(v29+v20+v30)
        v43 = constants[61]*c41*c44-constants[62]*c46
        v42 = constants[59]*c44*c45-constants[60]*c46
        dy_dt_45 = v42+v43
        v44 = constants[63]*c47*c45-constants[64]*c48
        v45 = constants[65]*c49*c45-constants[66]*c48
        dy_dt_47 = v44+v45
        v46 = constants[63]*c49*c45-constants[64]*c50
        v47 = constants[67]*c51*c45-constants[68]*c50
        dy_dt_44 = -(v29+v42+v44+v45+v46+v47)
        dy_dt_49 = v46+v47
        v49 = constants[71]*c49*c53-constants[72]*c52
        v48 = constants[69]*c51*c53-constants[70]*c52
        dy_dt_51 = v48+v49
        v50 = constants[73]*c53*c49-constants[74]*c54
        dy_dt_48 = -(v45+v46+v49+v50)
        v51 = constants[71]*c47*c53-constants[72]*c54
        dy_dt_53 = v50+v51
        v52 = constants[75]*c55*c51-constants[76]*c56
        v53 = constants[77]*c51*c57-constants[78]*c56
        dy_dt_55 = v52+v53
        v54 = constants[75]*c51*c57-constants[76]*c58
        v55 = constants[79]*c59*c51-constants[80]*c58
        dy_dt_50 = -(v47+v53+v55+v48+v52+v54)
        dy_dt_57 = v54+v55
        v57 = constants[83]*c57*c60-constants[84]*c61
        v56 = constants[81]*c59*c60-constants[82]*c61
        dy_dt_60 = v56+v57
        v58 = constants[85]*c60*c57-constants[86]*c62
        dy_dt_56 = -(v53+v54+v57+v58)
        v59 = constants[83]*c55*c60-constants[84]*c62
        dy_dt_61 = v58+v59
        v62 = constants[87]*c8-constants[88]*c87
        dy_dt_7 = (v7+v12)-(v14+v62)
        v22 = constants[32]*c31*c15-constants[33]*c32
        v69 = constants[32]*c31*c17-constants[33]*c63
        v36 = constants[51]*c40-constants[52]*c31
        dy_dt_30 = v36-(v22+v69)
        v63 = constants[20]*c17*c22-constants[21]*c18
        v24 = constants[20]*c22*c33-constants[36]*c34
        v35 = constants[49]*c24*c22-constants[50]*c30
        v38 = constants[20]*c22*c40-constants[36]*c39
        v71 = constants[20]*c22*c64-constants[36]*c65
        dy_dt_21 = -(v16+v24+v35+v38+v63+v71)
        v66 = constants[26]*c69*c19-constants[27]*c20
        v75 = constants[39]*c69*c41-constants[40]*c70
        v74 = constants[26]*c66*c69-constants[27]*c67
        dy_dt_68 = -(v66+v74+v75)
        v76 = constants[41]*c71*c72-constants[42]*c70
        dy_dt_69 = v75+v76
        v67 = constants[28]*c71*c19-constants[29]*c21
        v77 = constants[28]*c71*c66-constants[29]*c68
        dy_dt_70 = -(v67+v76+v77)
        v65 = constants[24]*c26*c19-constants[25]*c20
        v68 = constants[30]*c19*c26-constants[31]*c21
        v18 = constants[24]*c26*c25-constants[25]*c27
        v21 = constants[30]*c25*c26-constants[31]*c29
        v26 = constants[24]*c26*c35-constants[25]*c36
        v31 = constants[30]*c35*c26-constants[31]*c37
        v78 = constants[30]*c66*c26-constants[31]*c68
        v73 = constants[24]*c26*c66-constants[25]*c67
        dy_dt_25 = -(v18+v21+v26+v31+v65+v68+v73+v78)
        v32 = constants[43]*c38*c15-constants[44]*c35
        v79 = constants[43]*c17*c38-constants[44]*c66
        v40 = constants[55]*c24*c39-constants[56]*c38
        v33 = constants[45]*c40*c30-constants[46]*c38
        dy_dt_37 = (v33+v40)-(v32+v79)
        v37 = constants[53]*c15*c40-constants[54]*c33
        v81 = constants[53]*c17*c40-constants[54]*c64
        dy_dt_39 = -(v33+v36+v37+v38+v81)
        v39 = constants[53]*c15*c39-constants[54]*c34
        v82 = constants[53]*c17*c39-constants[54]*c65
        dy_dt_38 = v38-(v39+v40+v82)
        v34 = constants[47]*c15*c30-constants[48]*c25
        v80 = constants[47]*c17*c30-constants[48]*c19
        v41 = constants[57]*c30*c33-constants[58]*c35
        v83 = constants[57]*c30*c64-constants[58]*c66
        dy_dt_29 = v35-(v33+v34+v41+v80+v83)
        v85 = constants[61]*c41*c44-constants[62]*c73
        dy_dt_40 = -(v28+v43+v75+v85)
        v84 = constants[59]*c44*c72-constants[60]*c73
        dy_dt_43 = -(v42+v43+v84+v85)
        dy_dt_72 = v84+v85
        v86 = constants[63]*c47*c72-constants[64]*c74
        v87 = constants[65]*c75*c72-constants[66]*c74
        dy_dt_73 = v86+v87
        v88 = constants[63]*c72*c75-constants[64]*c76
        v89 = constants[67]*c72*c77-constants[68]*c76
        dy_dt_71 = -(v76+v84+v86+v87+v88+v89)
        dy_dt_75 = v88+v89
        v90 = constants[69]*c77*c53-constants[70]*c78
        v91 = constants[71]*c75*c53-constants[72]*c78
        dy_dt_77 = v90+v91
        v92 = constants[73]*c53*c75-constants[74]*c79
        dy_dt_74 = -(v87+v88+v91+v92)
        v94 = constants[75]*c55*c77-constants[76]*c80
        v95 = constants[77]*c81*c77-constants[78]*c80
        dy_dt_79 = v94+v95
        v93 = constants[71]*c47*c53-constants[72]*c79
        dy_dt_46 = -(v44+v51+v86+v93)
        dy_dt_52 = -(v48+v49+v50+v51+v90+v91+v92+v93)
        dy_dt_78 = v92+v93
        v96 = constants[75]*c77*c81-constants[76]*c82
        v97 = constants[79]*c83*c77-constants[80]*c82
        dy_dt_76 = -(v89+v90+v94+v95+v96+v97)
        dy_dt_81 = v96+v97
        v102 = constants[10]*c15-constants[11]*c17
        dy_dt_14 = v8-(v16+v22+v32+v34+v37+v39+v102)
        v98 = constants[81]*c83*c60-constants[82]*c84
        v99 = constants[83]*c81*c60-constants[84]*c84
        dy_dt_83 = v98+v99
        v23 = constants[34]*c32-constants[35]*c33
        v103 = constants[10]*c32-constants[11]*c63
        dy_dt_31 = v22-(v23+v103)
        v100 = constants[85]*c60*c81-constants[86]*c85
        dy_dt_80 = -(v95+v96+v99+v100)
        v104 = constants[10]*c33-constants[11]*c64
        dy_dt_32 = (v23+v37)-(v24+v41+v104)
        v101 = constants[83]*c55*c60-constants[84]*c85
        dy_dt_54 = -(v52+v59+v94+v101)
        dy_dt_59 = -(v56+v57+v58+v59+v98+v99+v100+v101)
        dy_dt_84 = v100+v101
        v107 = constants[8]*c9*c19-constants[9]*c88
        v106 = constants[6]*c25*c12-constants[7]*c88
        dy_dt_87 = v106+v107
        v109 = constants[6]*c27*c12-constants[7]*c89
        v108 = constants[10]*c27-constants[11]*c20
        dy_dt_26 = (v18+v19)-(v108+v109)
        v110 = constants[8]*c9*c20-constants[9]*c89
        dy_dt_88 = v109+v110
        v112 = constants[6]*c29*c12-constants[7]*c90
        v111 = constants[10]*c29-constants[11]*c21
        dy_dt_28 = (v20+v21)-(v111+v112)
        v113 = constants[8]*c9*c21-constants[9]*c90
        dy_dt_89 = v112+v113
        v115 = constants[6]*c34*c12-constants[7]*c91
        v25 = constants[37]*c24*c34-constants[38]*c35
        v114 = constants[10]*c34-constants[11]*c65
        dy_dt_33 = (v39+v24)-(v25+v114+v115)
        v116 = constants[8]*c9*c65-constants[9]*c91
        dy_dt_90 = v115+v116
        v119 = constants[8]*c9*c66-constants[9]*c92
        v118 = constants[6]*c35*c12-constants[7]*c92
        dy_dt_91 = v118+v119
        v121 = constants[6]*c36*c12-constants[7]*c93
        v120 = constants[10]*c36-constants[11]*c67
        dy_dt_35 = (v27+v26)-(v120+v121)
        v122 = constants[8]*c9*c67-constants[9]*c93
        dy_dt_92 = v121+v122
        v15 = constants[18]*c9-constants[19]*c12
        v124 = constants[6]*c37*c12-constants[7]*c94
        dy_dt_11 = v15-(v4+v106+v109+v112+v115+v118+v121+v124)
        v123 = constants[10]*c37-constants[11]*c68
        dy_dt_36 = (v30+v31)-(v123+v124)
        v125 = constants[8]*c68*c9-constants[9]*c94
        dy_dt_8 = -(v5+v15+v107+v110+v113+v116+v119+v122+v125)
        dy_dt_93 = v124+v125
        v105 = constants[10]*c25-constants[11]*c19
        v126 = constants[91]*c59*c25-constants[92]*c95
        dy_dt_24 = (v17+v34)-(v18+v19+v20+v21+v105+v106+v126)
        v117 = constants[10]*c35-constants[11]*c66
        v128 = constants[91]*c59*c35-constants[92]*c97
        dy_dt_34 = (v25+v32+v41)-(v117+v118+v128+v26+v27+v30+v31)
        v132 = constants[87]*c17-constants[88]*c87
        dy_dt_16 = (v14+v102)-(v63+v69+v79+v80+v81+v82+v132)
        v64 = constants[22]*c24*c18-constants[23]*c19
        v133 = constants[87]*c18-constants[88]*c87
        dy_dt_17 = (v9+v63)-(v5+v64+v133)
        v72 = constants[37]*c24*c65-constants[38]*c66
        v130 = constants[91]*c59*c24-constants[92]*c101
        v131 = constants[91]*c83*c24-constants[92]*c102
        dy_dt_23 = -(v17+v25+v35+v40+v64+v72+v130+v131)
        v127 = constants[91]*c83*c19-constants[92]*c96
        v134 = constants[87]*c19-constants[88]*c87
        dy_dt_18 = (v80+v64+v105)-(v65+v66+v67+v68+v107+v127+v134)
        v143 = constants[93]*c59*c99-constants[94]*c95
        dy_dt_94 = v126+v143
        v135 = constants[87]*c20-constants[88]*c87
        dy_dt_19 = (v65+v66+v108)-(v110+v135)
        v144 = constants[93]*c59*c99-constants[94]*c97
        dy_dt_96 = v128+v144
        dy_dt_98 = -(v143+v144)
        v136 = constants[87]*c21-constants[88]*c87
        dy_dt_20 = (v67+v68+v111)-(v113+v136)
        v145 = constants[93]*c59*c103-constants[94]*c101
        dy_dt_58 = -(v55+v56+v126+v128+v130+v143+v144+v145)
        dy_dt_100 = v130+v145
        v70 = constants[34]*c63-constants[35]*c64
        v137 = constants[87]*c63-constants[88]*c87
        dy_dt_62 = (v69+v103)-(v70+v137)
        v146 = constants[93]*c83*c100-constants[94]*c96
        dy_dt_95 = v127+v146
        v138 = constants[87]*c64-constants[88]*c87
        dy_dt_63 = (v70+v81+v104)-(v71+v83+v138)
        v129 = constants[91]*c83*c66-constants[92]*c98
        v147 = constants[93]*c83*c100-constants[94]*c98
        dy_dt_97 = v129+v147
        dy_dt_99 = -(v146+v147)
        v139 = constants[87]*c65-constants[88]*c87
        dy_dt_64 = (v71+v82+v114)-(v72+v116+v139)
        v148 = constants[93]*c83*c103-constants[94]*c102
        dy_dt_82 = -(v97+v98+v127+v129+v131+v146+v147+v148)
        dy_dt_101 = v131+v148
        dy_dt_102 = -(v145+v148)
        v140 = constants[87]*c66-constants[88]*c87
        dy_dt_65 = (v72+v79+v83+v117)-(v73+v74+v77+v78+v119+v129+v140)
        v141 = constants[87]*c67-constants[88]*c87
        dy_dt_66 = (v73+v74+v120)-(v122+v141)
        v142 = constants[87]*c68-constants[88]*c87
        dy_dt_67 = (v77+v78+v123)-(v125+v142)
        dy_dt_86 = v62+v132+v133+v134+v135+v136+v137+v138+v139+v140+v141+v142
        
        return (dy_dt_0, dy_dt_1, dy_dt_2, dy_dt_3, dy_dt_4, dy_dt_5, dy_dt_6,
                    dy_dt_7, dy_dt_8, dy_dt_9, dy_dt_10, dy_dt_11, dy_dt_12,
                    dy_dt_13, dy_dt_14, dy_dt_15, dy_dt_16, dy_dt_17, dy_dt_18,
                    dy_dt_19, dy_dt_20, dy_dt_21, dy_dt_22, dy_dt_23, dy_dt_24,
                    dy_dt_25, dy_dt_26, dy_dt_27, dy_dt_28, dy_dt_29, dy_dt_30,
                    dy_dt_31, dy_dt_32, dy_dt_33, dy_dt_34, dy_dt_35, dy_dt_36,
                    dy_dt_37, dy_dt_38, dy_dt_39, dy_dt_40, dy_dt_41, dy_dt_42,
                    dy_dt_43, dy_dt_44, dy_dt_45, dy_dt_46, dy_dt_47, dy_dt_48,
                    dy_dt_49, dy_dt_50, dy_dt_51, dy_dt_52, dy_dt_53, dy_dt_54,
                    dy_dt_55, dy_dt_56, dy_dt_57, dy_dt_58, dy_dt_59, dy_dt_60,
                    dy_dt_61, dy_dt_62, dy_dt_63, dy_dt_64, dy_dt_65, dy_dt_66,
                    dy_dt_67, dy_dt_68, dy_dt_69, dy_dt_70, dy_dt_71, dy_dt_72,
                    dy_dt_73, dy_dt_74, dy_dt_75, dy_dt_76, dy_dt_77, dy_dt_78,
                    dy_dt_79, dy_dt_80, dy_dt_81, dy_dt_82, dy_dt_83, dy_dt_84,
                    dy_dt_85, dy_dt_86, dy_dt_87, dy_dt_88, dy_dt_89, dy_dt_90,
                    dy_dt_91, dy_dt_92, dy_dt_93, dy_dt_94, dy_dt_95, dy_dt_96,
                    dy_dt_97, dy_dt_98, dy_dt_99, dy_dt_100, dy_dt_101, dy_dt_102,)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        y0_dict = {'c1':5e-08, # this is Molar (moles/L)
        'c2':50000.0,
        'c3':0.0,
        'c4':0.0,
        'c5':0.0,
        'c6':0.0,
        'c7':0.0,
        'c8':0.0,
        'c9':0.0,
        'c10':0.0,
        'c11':0.0,
        'c12':81000.0,
        'c13':0.0,
        'c14':12000.0,
        'c15':0.0,
        'c16':0.0,
        'c17':0.0,
        'c18':0.0,
        'c19':0.0,
        'c20':0.0,
        'c21':0.0,
        'c22':11000.0,
        'c23':0.0,
        'c24':26300.0,
        'c25':0.0,
        'c26':72000.0,
        'c27':0.0,
        'c28':0.0,
        'c29':0.0,
        'c30':40000.0,
        'c31':101000.0,
        'c32':0.0,
        'c33':0.0,
        'c34':0.0,
        'c35':0.0,
        'c36':0.0,
        'c37':0.0,
        'c38':0.0,
        'c39':0.0,
        'c40':0.0,
        'c41':40000.0,
        'c42':0.0,
        'c43':0.0,
        'c44':40000.0,
        'c45':0.0,
        'c46':0.0,
        'c47':21000000.0,
        'c48':0.0,
        'c49':0.0,
        'c50':0.0,
        'c51':0.0,
        'c52':0.0,
        'c53':40000.0,
        'c54':0.0,
        'c55':22100000.0,
        'c56':0.0,
        'c57':0.0,
        'c58':0.0,
        'c59':0.0,
        'c60':10000000.0,
        'c61':0.0,
        'c62':0.0,
        'c63':0.0,
        'c64':0.0,
        'c65':0.0,
        'c66':0.0,
        'c67':0.0,
        'c68':0.0,
        'c69':0.0,
        'c70':0.0,
        'c71':0.0,
        'c72':0.0,
        'c73':0.0,
        'c74':0.0,
        'c75':0.0,
        'c76':0.0,
        'c77':0.0,
        'c78':0.0,
        'c79':0.0,
        'c80':0.0,
        'c81':0.0,
        'c82':0.0,
        'c83':0.0,
        'c84':0.0,
        'c85':0.0,
        'c86':0.0,
        'c87':0.0,
        'c88':0.0,
        'c89':0.0,
        'c90':0.0,
        'c91':0.0,
        'c92':0.0,
        'c93':0.0,
        'c94':0.0,
        'c95':0.0,
        'c96':0.0,
        'c97':0.0,
        'c98':0.0,
        'c99':0.0,
        'c100':0.0,
        'c101':0.0,
        'c102':0.0,
        'c103':0.0,}

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup
    
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        param_dict = {'k1': 30000000.0,
            'kd1': 0.00384,
            'k2': 1.66e-05,
            'kd2': 0.1,
            'k3': 1.0,
            'kd3': 0.01,
            'k4': 1.73e-07,
            'kd4': 0.00166,
            'k5': 0.0,
            'kd5': 0.0146,
            'k6': 0.0005,
            'kd6': 0.005,
            'k8': 1.66e-06,
            'kd8': 0.2,
            'k10b': 0.0543,
            'kd10': 0.011,
            'k13': 2.17,
            'kd13': 0.0,
            'k15': 10000.0,
            'kd15': 0.0,
            'k16': 1.66e-05,
            'kd63': 0.275,
            'k17': 1.66e-05,
            'kd17': 0.06,
            'k18': 2.5e-05,
            'kd18': 1.3,
            'k19': 1.66e-07,
            'kd19': 0.5,
            'k20': 3.5e-06,
            'kd20': 0.4,
            'k21': 3.66e-07,
            'kd21': 0.023,
            'k22': 3.5e-05,
            'kd22': 0.1,
            'k23': 6.0,
            'kd23': 0.06,
            'kd24': 0.55,
            'k25': 1.66e-05,
            'kd25': 0.0214,
            'k28': 1.66e-06,
            'kd28': 0.0053,
            'k29': 1.17e-06,
            'kd29': 1.0,
            'k32': 4e-07,
            'kd32': 0.1,
            'k33': 3.5e-05,
            'kd33': 0.2,
            'k34': 7.5e-06,
            'kd34': 0.03,
            'k35': 7.5e-06,
            'kd35': 0.0015,
            'k36': 0.005,
            'kd36': 0.0,
            'k37': 1.5e-06,
            'kd37': 0.3,
            'k40': 5e-05,
            'kd40': 0.064,
            'k41': 5e-05,
            'kd41': 0.0429,
            'k42': 0.000118,
            'kd42': 0.2,
            'k43': 0.0,
            'kd43': 1.0,
            'k44': 1.95e-05,
            'kd52': 0.033,
            'k45': 0.0,
            'kd45': 3.5,
            'k47': 0.0,
            'kd47': 2.9,
            'k48': 2.38e-05,
            'kd48': 0.8,
            'k49': 0.0,
            'kd49': 0.0568,
            'k50': 4.5e-07,
            'kd50': 0.5,
            'k52': 8.91e-05,
            'kd44': 0.0183,
            'k53': 0.0,
            'kd53': 16.0,
            'k55': 0.0,
            'kd55': 5.7,
            'k56': 2.35e-05,
            'kd56': 0.6,
            'k57': 0.0,
            'kd57': 0.246,
            'k58': 8.33e-06,
            'kd58': 0.5,
            'k60': 0.0055,
            'kd60': 0.0,
            'k61': 0.00067,
            'kd61': 0.0,
            'k126': 1.66e-07,
            'kd126': 2.0,
            'k127': 0.0,
            'kd127': 0.0001,
            'constants[95]': 0.0}

        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

