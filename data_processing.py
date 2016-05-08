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
                   COMPL_RPY_5YR_RT, NONCOM_RPY_5YR_RT, LO_INC_RPY_5YR_RT, MD_INC_RPY_5YR_RT, HI_INC_RPY_5YR_RT, DEP_RPY_5YR_RT, IND_RPY_5YR_RT, \
                   PELL_RPY_5YR_RT, NOPELL_RPY_5YR_RT, FEMALE_RPY_5YR_RT, MALE_RPY_5YR_RT, FIRSTGEN_RPY_5YR_RT, NOTFIRSTGEN_RPY_5YR_RT, RPY_7YR_RT,\
                   COMPL_RPY_7YR_RT, NONCOM_RPY_7YR_RT, LO_INC_RPY_7YR_RT, MD_INC_RPY_7YR_RT, HI_INC_RPY_7YR_RT, DEP_RPY_7YR_RT, IND_RPY_7YR_RT, \
                   PELL_RPY_7YR_RT, NOPELL_RPY_7YR_RT, FEMALE_RPY_7YR_RT, MALE_RPY_7YR_RT, FIRSTGEN_RPY_7YR_RT, NOTFIRSTGEN_RPY_7YR_RT, count_ed,\
                   count_nwne_p6, count_wne_p6, mn_earn_wne_p6, md_earn_wne_p6, pct10_earn_wne_p6, pct25_earn_wne_p6, pct75_earn_wne_p6, \
                   pct90_earn_wne_p6, sd_earn_wne_p6, gt_25k_p6, mn_earn_wne_inc1_p6, mn_earn_wne_inc2_p6, mn_earn_wne_inc3_p6 from Scorecard', con)

# extract 2005 and 2013 data
raw_2013 = raw.ix[raw.Year==2013,:]
raw_2005 = raw.ix[raw.Year==2005,:]
raw_2013 = raw_2013.reset_index()
raw_2005 = raw_2005.reset_index()
raw_2013 = raw_2013.drop('index',axis=1)
raw_2005 = raw_2005.drop('index',axis=1)

# find schools that exist in 2013 but not 2005 
id_t = []
id_special = []
nonnull_counts = raw_2013.count()
for i in range(len(nonnull_counts)):
    if float(nonnull_counts.ix[i,0])/float(len(raw_2013)) < 0.7:
        if float(nonnull_counts.ix[i,0])/float(len(raw_2013)) > 0:
            id_special.append(i)
        else:
            id_t.append(i)
# drop empty feature columns from 2013 data 
data_2013 = raw_2013.drop(raw_2013.columns[id_t],axis=1)
# extract corresponding columns from 2005 data
data_2005 = raw_2005.ix[:,id_t]
# set up lookup matrix from UNITID
id_t.append(1)
data_2005_lookup = raw_2005.ix[:,id_t]
id_t.remove(1)
# create new columns of accurate size to append
additional_cols = pd.DataFrame(-1,index=np.arange(len(data_2013)),columns=data_2005.columns)
data_full = data_2013.join(additional_cols) # join columns
data_full.shape # sanity check
# replace values in 2013 with 2005; nonexistant schools in 2005 remain as -1
for i in range(0,len(data_2013)):
    find_id = data_2013.ix[i,'UNITID']
    for j in range(0,len(data_2005)):
        if data_2005_lookup.ix[j,'UNITID']==find_id:
            for k in data_2005.columns:
                data_full.ix[i,k] = data_2005.ix[j,k]

# filter out online schools and schools that don't offer at least bachelor's degrees
data_filtered = []
for i in range(0, len(data)):
    if (data.ix[i, 'DISTANCEONLY'] == "Not distance-education only") and \
        ((data.ix[i, 'HIGHDEG'] == "Graduate degree") or (data.ix[i, 5] == "Bachelor's degree")) and \
        (data.ix[i, 'CURROPER'] == "Currently certified as operating"):
        data_filtered.append(data.ix[i])
# create dataframe with data_filtered
data_final = pd.DataFrame(data_filtered, columns=data.columns.values)

# drop categorical data and data superfluous to regression
data_final = data_final.drop(['DISTANCEONLY', 'HIGHDEG', 'CURROPER', 'INSTNM', 'LATITUDE', 'LONGITUDE', 'ZIP', 'STABBR', 'region', 'LOCALE'], axis=1) 
# reset indices
data_final = data_final.reset_index()
data_final = data_final.drop('index',axis=1)

# drop schools that don't exist in 2005 (-1)
to_drop = []
for i in range(len(data_final)): 
    for j in range(0, len(data_final.columns)):
        if data_final.loc[i, j] == -1:
            to_drop.append(i)
            break
data_processed = data_final.drop(data_final.index[to_drop])
data_processed = data_processed.reset_index()
data_processed = data_processed.drop('index',axis=1)
# cast PrivacySuppressed into NaNs for easier imputation later
data_processed = data_processed.replace('PrivacySuppressed',np.nan)

# remove redundant columns from data
data_processed = data_processed.drop(['Year, SATWR25', 'SATWR75', 'SATWRMID', 'ACTWR25', 'ACTWR75', 'ACTWRMID'], axis = 1)
data_processed = data_processed.drop(['RPY_3YR_RT', 'COMPL_RPY_3YR_RT', 'NONCOM_RPY_3YR_RT', 'LO_INC_RPY_3YR_RT', \
                                      'MD_INC_RPY_3YR_RT', 'HI_INC_RPY_3YR_RT', 'DEP_RPY_3YR_RT', 'IND_RPY_3YR_RT', \
                                      'PELL_RPY_3YR_RT', 'NOPELL_RPY_3YR_RT', 'FEMALE_RPY_3YR_RT', 'MALE_RPY_3YR_RT', \
                                      'FIRSTGEN_RPY_3YR_RT', 'NOTFIRSTGEN_RPY_3YR_RT', 'RPY_7YR_RT', 'COMPL_RPY_7YR_RT', \
                                      'NONCOM_RPY_7YR_RT', 'LO_INC_RPY_7YR_RT', 'MD_INC_RPY_7YR_RT', 'HI_INC_RPY_7YR_RT', \
                                      'DEP_RPY_7YR_RT', 'IND_RPY_7YR_RT', 'PELL_RPY_7YR_RT', 'NOPELL_RPY_7YR_RT', \
                                      'FEMALE_RPY_7YR_RT', 'MALE_RPY_7YR_RT', 'FIRSTGEN_RPY_7YR_RT', \
                                      'NOTFIRSTGEN_RPY_7YR_RT'], axis=1)
data_remove_e = data_processed.drop(['count_ed', 'count_nwne_p6', 'count_wne_p6', 'md_earn_wne_p6', 'pct10_earn_wne_p6',\
                                     'pct25_earn_wne_p6', 'pct75_earn_wne_p6', 'pct90_earn_wne_p6', 'sd_earn_wne_p6', \
                                     'gt_25k_p6', 'mn_earn_wne_inc1_p6', 'mn_earn_wne_inc2_p6',\
                                     'mn_earn_wne_inc3_p6'], axis=1)

# remove schools that have more than 90% null values
nan_counts = data_remove_rpl.count(axis=1)
proportion_nan = []
proportion_nan[:] = [1 - ((float) (a) / (float) (len(data_remove_rpl.columns))) for a in nan_counts]
df_p_nan = pd.DataFrame(proportion_nan)
delete_these_rows = df_p_nan.ix[df_p_nan[0] > 0.9].index.values
data_thinner = data_remove_rpl.drop(data_remove_rpl.index[delete_these_rows])
data_thinner = data_thinner.reset_index() # reset indices again
data_thinner = data_thinner.drop('index', axis=1)
keep_indices = np.isfinite(data_thinner.mn_earn_wne_p6)
drop_this = []
for i in range(len(keep_indices)):
    if keep_indices.ix[i] == False:
        drop_this.append(i)
data_super_thin = data_thinner.drop(data_thinner.index[drop_this])

# MATLAB deals with NaNs better as a string 'nan'
data_super_nans = data_super_thin.replace(np.nan,'nan')
data_super_nans.to_csv('data_super_nan.csv', encoding='utf-8')

###----------MATLAB CODE FOR IMPUTATION----------###
### data = csvread('data_super_nan.csv',1,1);
### data_imputed = knnimpute(data');
### data_imputed = data_imputed';
### dlmwrite('data_imputed.csv',data_imputed,'precision','%6.6f')
###----------END MATLAB CODE FOR IMPUTATION----------###

# read in MATLAB output
data_needs_headers = pd.read_csv("data_imputed.csv",low_memory=False,header=None)
data_super_nans = data_super_nans.drop('level_0', axis=1)
data_still_needs_headers = data_needs_headers.drop(0,axis=1)
# set columns equal to pre-MATLAB imputation values
data_still_needs_headers.columns = data_super_nans.columns

# save full data set
data_still_needs_headers.to_csv('data_full_final.csv', encoding='utf-8')

# normalize the data using z-scores
data_z = stats.zscore(data_still_needs_headers, axis=1)
data_z_full = pd.DataFrame(data=data_z,columns=data_super_nans.columns)
# make sure UNITID remains un-normalized
data_z_full.ix[:,'UNITID'] = data_still_needs_headers.ix[:,'UNITID']

# save z-scored full data set
data_z_full.to_csv('data_z_full_final.csv', encoding='utf-8')

# separate into x and y data for regression
y_full = data_still_needs_headers.ix[:,'mn_earn_wne_p6']
y_full_z = data_z_full.ix[:,'mn_earn_wne_p6']
x_full = data_still_needs_headers.drop('mn_earn_wne_p6', axis=1)
x_full_z = data_z_full.drop('mn_earn_wne_p6', axis=1)

# save x and y data both original and z-scored
y_full.to_csv('y_full.csv', encoding='utf-8',index=False)
y_full_z.to_csv('y_full_z.csv', encoding='utf-8',index=False)
x_full.to_csv('x_full.csv', encoding='utf-8',index=False)
x_full_z.to_csv('x_full_z.csv', encoding='utf-8',index=False)