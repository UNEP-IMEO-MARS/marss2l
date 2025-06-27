offshore = "MEX_003,MEX_004,MEX_005,MYS_1"

OFFSHORE = [o.strip() for o in offshore.split(",")]

turkmenistan = """EMIT_0029,T_5,T_12,T_20,EMIT_0045,T_9,T_10,T_11,T_7,ZY1_S_11,Te_M_14,T_22,T_24,
                  ZY1_S_2,T_27,Te_M_8,Tc_7_1,Te_N_8,T_14,Te_14_1,T_25,Te_M_1,T_6,T_19,Te_M_13,Tc_7_6,
                  T_26,Te_M_7,T_28,EMIT_0059,Te_N_1,T_GS_01,T_23,T_EMIT_502,EMIT_CH4_PlumeComplex-113,
                  T_EMIT_436,EMIT_CH4_PlumeComplex-1705,D.1,Te_M_15,T_EMIT_114,T_EMIT_0314,EMIT_0329,
                  EMIT_0382,T_EMIT_0019,EMIT_0373,EMIT_0385,T_17,T_EMIT_1720,T_EMIT_1721,T_EMIT_1059,
                  EMIT_0220,T_1,T_0,T_13,Te_M_12,T_4,T_EMIT_1048,T_EMIT_133,EMIT_0221,Te_N_10,T_EMIT_2047,
                  Te_M_2,Te_N_9,Te_M_5,Te_M_10,Tc_7_3,ZY1_S_13,ZY1_S_7,Tc_7_2,Te_N_12,ZY1_S_9,T_EMIT_13,
                  T_GS_02,Te_N_3,T_GS_03,ZY1_S_12,ZY1_S_5,Te_N_7,Te_M_4,Te_M_11,ZY1_S_6,ZY1_S_10,ZY1_S_1,
                  ZY1_S_8,Te_N_2,T_100,EMIT_CH4_PlumeComplex-530,EMIT_0386,EMIT_CH4_PlumeComplex-896,T_EMIT_227,
                  T_EMIT_6,T_EMIT_46,EMIT_CH4_PlumeComplex-26,Te_M_9,Tc_13_1,EMIT_0013,EMIT_CH4_PlumeComplex-50,
                  EMIT_CH4_PlumeComplex-1870,Te_N_13,EMIT_CH4_PlumeComplex-1869,TKM_lat36p506lon61p570,
                  TKM_lat36p568lon61p673,TKM_lat36p451lon61.588,TKM_lat36p514lon61.600,TKM_lat36p406lon61p558,
                  TKM_lat36p514lon61p539,T_EMIT_2048,Te_N_5,EMIT_0328,EMIT_0387"""

TURKMENISTAN = [t.strip() for t in turkmenistan.split(",")]
# T_EMIT_2097
permian = """EMIT_CH4_PlumeComplex-395,EMIT_CH4_PlumeComplex-1999,y,PB_50,PB_61,EMIT_CH4_PlumeComplex-353,
                  PB_E_01,PB_23,EMIT_CH4_PlumeComplex-1892,PB_33,PB_K_0003,PB_K_0004,PB_K_0005,EMIT_CH4_PlumeComplex-2267,
                  PB_lat32.2169lon-103p5230,PB_073,PB_K2,PB_29,EMIT_CH4_PlumeComplex-1994,ak/P_1,PB_63,PB_18,PB_19,
                  PB_EnMAP_1,PB_K_0001,PB_66,PB_67,PB_69,aa,PB_lat31p2896lon103p1544,EMIT_CH4_PlumeComplex-355,
                  PB_58,PB_28,PB_48,PB_70,EMIT_CH4_PlumeComplex-1897,PB_26,PB_16,PB_44,EMIT_CH4_PlumeComplex-2255,
                  EMIT_CH4_PlumeComplex-2259,EMIT_CH4_PlumeComplex-2261,EMIT_CH4_PlumeComplex-2266,EMIT_CH4_PlumeComplex-2477,
                  EMIT_CH4_PlumeComplex-2483,EMIT_CH4_PlumeComplex-2485,PB_1,ah,EMIT_CH4_PlumeComplex-1998,
                  EMIT_CH4_PlumeComplex-2264,PB_074,PB_075,PB_076,v,EMIT_CH4_PlumeComplex-2249,PB_54,PB_51,PB_47,
                  PB_45,PB_27,PB_30,PB_32,PB_34,PB_35,PB_49,PB_46,PB_53,PB_55,PB_56,PB_57,EMIT_CH4_PlumeComplex-1995,
                  PB_31,PB_17,w,PB_14,EMIT_CH4_PlumeComplex-2142,PB_22,PB_60,PB_24,PB_25,S2_x,EMIT_CH4_PlumeComplex-1531,
                  San_Juan,PB_13-GOES,EMIT_CH4_PlumeComplex-2486,PB_59,PB_62,EMIT_CH4_PlumeComplex-1532,PB_G_20,e,PB_21,
                  PB_lat32p2214lon-103p5291,PB_68,EMIT_0149,EMIT_CH4_PlumeComplex-965,EMIT_CH4_PlumeComplex-400,
                  EMIT_CH4_PlumeComplex-2139,EMIT_CH4_PlumeComplex-966,EMIT_CH4_PlumeComplex-963,EMIT_CH4_PlumeComplex-2141,
                  EMIT_CH4_PlumeComplex-2145,EMIT_CH4_PlumeComplex-2138,EMIT_CH4_PlumeComplex-406,EMIT_CH4_PlumeComplex-2146,
                  EMIT_CH4_PlumeComplex-2010,EMIT_0135,EMIT_CH4_PlumeComplex-2140,EMIT_CH4_PlumeComplex-1026,PB_6,
                  EMIT_CH4_PlumeComplex-1782,EMIT_CH4_PlumeComplex-1894,EMIT_CH4_PlumeComplex-1028,EMIT_CH4_PlumeComplex-1062,
                  EMIT_CH4_PlumeComplex-1108,EMIT_CH4_PlumeComplex-401,EMIT_CH4_PlumeComplex-1063,EMIT_CH4_PlumeComplex-405,
                  EMIT_CH4_PlumeComplex-1027,f,h,EMIT_20231020_miss,EMIT_CH4_PlumeComplex-342,EMIT_CH4_PlumeComplex-1029,EMIT_0041,
                  PB_EM_1,PB_EM_2,EMIT_CH4_PlumeComplex-1451,EMIT_CH4_PlumeComplex-1447,EMIT_CH4_PlumeComplex-1448,EMIT_CH4_PlumeComplex-1449,
                  EMIT_CH4_PlumeComplex-1450,EMIT_CH4_PlumeComplex-1452,EMIT_CH4_PlumeComplex-1037,EMIT_CH4_PlumeComplex-1038,
                  EMIT_CH4_PlumeComplex-1039,EMIT_CH4_PlumeComplex-1520,EMIT_CH4_PlumeComplex-1036,PB_7,c,PB_8,PB_9,PB_5,
                  EMIT_CH4_PlumeComplex-398,g,PB_12,i,EMIT_CH4_PlumeComplex-399,EMIT_CH4_PlumeComplex-403,EMIT_CH4_PlumeComplex-396,
                  EMIT_0148,EMIT_0153,EMIT_0150,EMIT_0155,EMIT_0246,EMIT_0247,EMIT_0152,EMIT_CH4_PlumeComplex-1529,EMIT_CH4_PlumeComplex-1534,
                  EMIT_CH4_PlumeComplex-1530,EMIT_CH4_PlumeComplex-394,EMIT_CH4_PlumeComplex-338,EMIT_CH4_PlumeComplex-350,
                  EMIT_CH4_PlumeComplex-346,EMIT_CH4_PlumeComplex-344,EMIT_CH4_PlumeComplex-343,EMIT_CH4_PlumeComplex-341,
                  EMIT_CH4_PlumeComplex-337,EMIT_CH4_PlumeComplex-335,EMIT_CH4_PlumeComplex-334,EMIT_CH4_PlumeComplex-333,
                  EMIT_CH4_PlumeComplex-331,PB_11,EMIT_CH4_PlumeComplex-345,n,PB_EnMAP_4,p,PB_EnMAP_3,PB_EnMAP_2,PB_EnMAP_5,
                  EMIT_0151,EMIT_0147,EMIT_CH4_PlumeComplex-66,EMIT_0145,EMIT_CH4_PlumeComplex-404,EMIT_CH4_PlumeComplex-221,
                  EMIT_CH4_PlumeComplex-219,EMIT_CH4_PlumeComplex-218,EMIT_CH4_PlumeComplex-220,EMIT_0339,EMIT_CH4_PlumeComplex-525,
                  EMIT_CH4_PlumeComplex-1109,PB_3,EMIT_CH4_PlumeComplex-1181,PB_2,EMIT_CH4_PlumeComplex-351,P_j_2,ZY1_snow_3,ZY1_snow_2,
                  ZY1_snow_1,GF5_02,S2_big,P_2,S2_emiss_07-10-2020,EMIT_CH4_PlumeComplex-1111,EMIT_0144,P_4,aj,ae,ad,t,q,ab,
                  EMIT_CH4_PlumeComplex-1022,PB_78,PB_81-82,PB_83"""

PERMIAN = [p.strip() for p in permian.split(",")]

CASE_STUDIES = {
    "Offshore": OFFSHORE,
    "Turkmenistan": TURKMENISTAN,
    "Permian": PERMIAN
}

REV_CASE_STUDIES = {vi: k for k, v in CASE_STUDIES.items() for vi in v}