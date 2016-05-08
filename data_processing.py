import numpy as np
from scipy import stats
import os
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import sqlite3
%matplotlib inline

con = sqlite3.connect('../data/database.sqlite')
raw = pd.read_sql('select Year, UNITID, INSTNM, DISTANCEONLY, HIGHDEG, UGDS, UGDS_WHITE, UGDS_BLACK, UGDS_HISP, UGDS_ASIAN, UGDS_AIAN, UGDS_NHPI,\
                   UGDS_2MOR, UGDS_NRA, UGDS_UNKN, UG25abv, INC_PCT_LO, DEP_STAT_PCT_IND, DEP_INC_PCT_LO, IND_INC_PCT_LO,\
                   DEP_INC_PCT_M1, DEP_INC_PCT_M2, DEP_INC_PCT_H1, IND_INC_PCT_M1, IND_INC_PCT_M2, IND_INC_PCT_H1, IND_INC_PCT_H2,\
                   DEP_INC_PCT_H2, PAR_ED_PCT_1STGEN, pct_white, pct_black, pct_asian, pct_hispanic, pct_ba, pct_grad_prof, \
                   pct_born_us, median_hh_inc, poverty_rate, unemp_rate, ln_median_hh_inc, pell_ever, female, \
                   fsend_1, fsend_2, fsend_3, fsend_4, fsend_5, INC_PCT_LO, INC_PCT_M1, INC_PCT_M2, INC_PCT_H1, INC_PCT_H2, LATITUDE,\
                   LONGITUDE, ZIP, STABBR, region, LOCALE, CURROPER, NPT4_PUB, NPT4_PRIV, NPT41_PUB, NPT42_PUB, NPT43_PUB, NPT44_PUB, \
                   NPT45_PUB, NPT41_PRIV, NPT42_PRIV, NPT43_PRIV, NPT44_PRIV, NPT45_PRIV, TUITIONFEE_IN, TUITIONFEE_OUT, PCTPELL, \
                   ADM_RATE, RET_FT4, SATVR25, SATVR75, SATMT25, SATMT75, SATWR25, SATWR75, SATVRMID, SATMTMID, SATWRMID, ACTCM25, \
                   ACTCM75, ACTEN25, ACTEN75, ACTMT25, ACTMT75, ACTWR25, ACTWR75, ACTCMMID, ACTENMID, ACTMTMID, ACTWRMID, SAT_AVG, \
                   SAT_AVG_ALL, PCIP01, PCIP03, PCIP04, PCIP05, PCIP09, PCIP10, PCIP11, PCIP12, PCIP13, PCIP14, PCIP15, PCIP16, PCIP19, \
                   PCIP22, PCIP23, PCIP24, PCIP25, PCIP26, PCIP27, PCIP29, PCIP30, PCIP31, PCIP38, PCIP39, PCIP40, PCIP41, PCIP42, PCIP43,\
                   PCIP44, PCIP45, PCIP46, PCIP47, PCIP48, PCIP49, PCIP50, PCIP51, PCIP52, PCIP54, C150_4_WHITE, C150_4_BLACK, C150_4_HISP, \
                   C150_4_ASIAN, ENRL_ORIG_YR2_RT, LO_INC_ENRL_ORIG_YR2_RT, MD_INC_ENRL_ORIG_YR2_RT, HI_INC_ENRL_ORIG_YR2_RT, DEP_ENRL_ORIG_YR2_RT,\
                   IND_ENRL_ORIG_YR2_RT, FEMALE_ENRL_ORIG_YR2_RT, MALE_ENRL_ORIG_YR2_RT, PELL_ENRL_ORIG_YR2_RT, NOPELL_ENRL_ORIG_YR2_RT,\
                   LOAN_ENRL_ORIG_YR2_RT, NOLOAN_ENRL_ORIG_YR2_RT, FIRSTGEN_ENRL_ORIG_YR2_RT, NOT1STGEN_ENRL_ORIG_YR2_RT, ENRL_ORIG_YR3_RT,\
                   LO_INC_ENRL_ORIG_YR3_RT, MD_INC_ENRL_ORIG_YR3_RT, HI_INC_ENRL_ORIG_YR3_RT, DEP_ENRL_ORIG_YR3_RT, IND_ENRL_ORIG_YR3_RT, \
                   FEMALE_ENRL_ORIG_YR3_RT, MALE_ENRL_ORIG_YR3_RT, PELL_ENRL_ORIG_YR3_RT, NOPELL_ENRL_ORIG_YR3_RT, LOAN_ENRL_ORIG_YR3_RT, \
                   NOLOAN_ENRL_ORIG_YR3_RT, FIRSTGEN_ENRL_ORIG_YR3_RT, NOT1STGEN_ENRL_ORIG_YR3_RT, ENRL_ORIG_YR4_RT, LO_INC_ENRL_ORIG_YR4_RT,\
                   MD_INC_ENRL_ORIG_YR4_RT, HI_INC_ENRL_ORIG_YR4_RT, DEP_ENRL_ORIG_YR4_RT, IND_ENRL_ORIG_YR4_RT, FEMALE_ENRL_ORIG_YR4_RT, \
                   MALE_ENRL_ORIG_YR4_RT, PELL_ENRL_ORIG_YR4_RT, NOPELL_ENRL_ORIG_YR4_RT, LOAN_ENRL_ORIG_YR4_RT, NOLOAN_ENRL_ORIG_YR4_RT, \
                   FIRSTGEN_ENRL_ORIG_YR4_RT, NOT1STGEN_ENRL_ORIG_YR4_RT, ENRL_ORIG_YR6_RT, LO_INC_ENRL_ORIG_YR6_RT, MD_INC_ENRL_ORIG_YR6_RT,\
                   HI_INC_ENRL_ORIG_YR6_RT, DEP_ENRL_ORIG_YR6_RT, IND_ENRL_ORIG_YR6_RT, FEMALE_ENRL_ORIG_YR6_RT, MALE_ENRL_ORIG_YR6_RT, \
                   PELL_ENRL_ORIG_YR6_RT, NOPELL_ENRL_ORIG_YR6_RT, LOAN_ENRL_ORIG_YR6_RT, NOLOAN_ENRL_ORIG_YR6_RT, FIRSTGEN_ENRL_ORIG_YR6_RT,\
                   NOT1STGEN_ENRL_ORIG_YR6_RT, ENRL_ORIG_YR8_RT, LO_INC_ENRL_ORIG_YR8_RT, MD_INC_ENRL_ORIG_YR8_RT, HI_INC_ENRL_ORIG_YR8_RT, \
                   DEP_ENRL_ORIG_YR8_RT, IND_ENRL_ORIG_YR8_RT, FEMALE_ENRL_ORIG_YR8_RT, MALE_ENRL_ORIG_YR8_RT, PELL_ENRL_ORIG_YR8_RT, \
                   NOPELL_ENRL_ORIG_YR8_RT, LOAN_ENRL_ORIG_YR8_RT, NOLOAN_ENRL_ORIG_YR8_RT, FIRSTGEN_ENRL_ORIG_YR8_RT, NOT1STGEN_ENRL_ORIG_YR8_RT,\
                   C150_4_POOLED_SUPP, PCTFLOAN, DEBT_MDN, GRAD_DEBT_MDN, WDRAW_DEBT_MDN, LO_INC_DEBT_MDN, MD_INC_DEBT_MDN, HI_INC_DEBT_MDN, \
                   DEP_DEBT_MDN, IND_DEBT_MDN, PELL_DEBT_MDN, NOPELL_DEBT_MDN, CUML_DEBT_N, CUML_DEBT_P90, CUML_DEBT_P75, CUML_DEBT_P25, CDR3, \
                   COMPL_RPY_1YR_RT, NONCOM_RPY_1YR_RT, LO_INC_RPY_1YR_RT, MD_INC_RPY_1YR_RT, HI_INC_RPY_1YR_RT, DEP_RPY_1YR_RT, IND_RPY_1YR_RT,\
                   PELL_RPY_1YR_RT, NOPELL_RPY_1YR_RT, FEMALE_RPY_1YR_RT, MALE_RPY_1YR_RT, FIRSTGEN_RPY_1YR_RT, NOTFIRSTGEN_RPY_1YR_RT, RPY_3YR_RT,\
                   COMPL_RPY_3YR_RT, NONCOM_RPY_3YR_RT, LO_INC_RPY_3YR_RT, MD_INC_RPY_3YR_RT, HI_INC_RPY_3YR_RT, DEP_RPY_3YR_RT, IND_RPY_3YR_RT,\
                   PELL_RPY_3YR_RT, NOPELL_RPY_3YR_RT, FEMALE_RPY_3YR_RT, MALE_RPY_3YR_RT, FIRSTGEN_RPY_3YR_RT, NOTFIRSTGEN_RPY_3YR_RT, RPY_5YR_RT,\
                   COMPL_RPY_5YR_RT, NONCOM_RPY_5YR_RT, LO_INC_RPY_5YR_RT, \
                   MD_INC_RPY_5YR_RT,\
                   HI_INC_RPY_5YR_RT, DEP_RPY_5YR_RT, IND_RPY_5YR_RT, PELL_RPY_5YR_RT, NOPELL_RPY_5YR_RT, \
                   FEMALE_RPY_5YR_RT,\
                   MALE_RPY_5YR_RT, FIRSTGEN_RPY_5YR_RT, NOTFIRSTGEN_RPY_5YR_RT, RPY_7YR_RT, COMPL_RPY_7YR_RT, \
                   NONCOM_RPY_7YR_RT,\
                   LO_INC_RPY_7YR_RT, MD_INC_RPY_7YR_RT, HI_INC_RPY_7YR_RT, DEP_RPY_7YR_RT, IND_RPY_7YR_RT, \
                   PELL_RPY_7YR_RT,\
                   NOPELL_RPY_7YR_RT, FEMALE_RPY_7YR_RT, MALE_RPY_7YR_RT, FIRSTGEN_RPY_7YR_RT, \
                   NOTFIRSTGEN_RPY_7YR_RT,\
                   count_ed,\
                   count_nwne_p6, count_wne_p6, mn_earn_wne_p6, md_earn_wne_p6, pct10_earn_wne_p6, \
                   pct25_earn_wne_p6,\
                   pct75_earn_wne_p6, pct90_earn_wne_p6, sd_earn_wne_p6, gt_25k_p6, mn_earn_wne_inc1_p6, \
                   mn_earn_wne_inc2_p6,\
                   mn_earn_wne_inc3_p6 from Scorecard', con)