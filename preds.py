import os
# Enforce CPU Usage
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncommenting enforces CPU usage  # Commenting enforces GPU usage

# Seed the Random-Number-Generator in a bid to get 'Reproducible Results'
import tensorflow as tf
from random import seed, sample
from numpy.random import seed 
seed(1)
tf.compat.v1.set_random_seed(3)

# load required modules
import pandas as pd
import numpy as np
import math, time
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, Normalizer
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, BayesianRidge, ARDRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Import classes from my custom package
from custom_classes.Starter_Module_01 import Starter

# Global settings for PANDAS frame display
pd.set_option('html.table_schema', True)
pd.set_option('max_columns', 800)
pd.set_option('max_rows', 70000)

 
def args_parse_cmd():
    parser = ArgumentParser(description='START-HELP: Program for forecasting/predicting breakup or schism in social networks', epilog='END-HELP: End of assistance/help section',
                            formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('-rp', '--root_path', nargs='+', default='datasets/', type=str, help='Generic root path for application/program')    
    parser.add_argument('-rm', '--run_mode', nargs='+', default='single', type=str, choices=['single', 'all'], help='Run model per specified dataset OR cumulatively for all intrinsic datasets')
    args = parser.parse_args()
    return args

def process_reprd_idx(myCls, root_path, fname):    
    df1 = myCls.load_data(root_path, fname, sep=',', header=0, index_col=0, mode='READ')
    exp_reprd_idx = pd.DataFrame()
    for i in range(len(df1)):
        tmp_start_date = df1.iloc[i,1]
        start_date = datetime.strptime(tmp_start_date, '%d-%m-%Y')
        tmp_end_date = df1.iloc[i,2]
        end_date = datetime.strptime(tmp_end_date, '%d-%m-%Y')
        
        while (start_date <= end_date):
            exp_reprd_idx = exp_reprd_idx.append([[str(start_date), df1.values[i,3]]], ignore_index=True)
            start_date = start_date + timedelta(days=1)            
        
    exp_reprd_idx.to_csv(root_path+fname[:-4]+'_EXPAND.csv', sep=',', header=False, index=False)
    
def cumm_2_reg_conv(myCls, root_path, fname):
    #df1 = myCls.load_data(root_path, fname, sep='\s', header=0, index_col=None, mode='READ')
    df1 = myCls.load_data(root_path, fname, sep=',', header=0, index_col=None, mode='READ')
    infect_recover_death = pd.DataFrame()
    for i in range(len(df1)):
        if (i == 0):
            temp_infect = df1.values[i,2]
            temp_recover = df1.values[i,5]
            temp_death = df1.values[i,6]
        else:
            temp_infect = df1.values[i,2] - df1.values[i-1,2]
            temp_recover = df1.values[i,5] - df1.values[i-1,5]
            temp_death = df1.values[i,6] - df1.values[i-1,6]
        infect_recover_death = infect_recover_death.append([[temp_infect, temp_recover, temp_death]], ignore_index=True)
        
    infect_recover_death.columns = ['Infected Positive (I)', 'Recovered (R)', 'Deaths (D)']
    infect_recover_death.to_csv(root_path+fname[:-4]+'_infect_recover_death.csv', sep=',', header=True, index=False)
    
def wkend_holiday(myCls, root_path, fname):
    df1 = myCls.load_data(root_path, fname, sep=',', header=0, index_col=None, mode='READ')
    holidays = pd.DataFrame()
    for i in range(len(df1)):
        if (i % 7 == 0):
            holidays = holidays.append([[1]], ignore_index=True)  # Saturdays
        elif ((i-1) % 7 == 0):
            holidays = holidays.append([[1]], ignore_index=True)  # Sundays
        else:
            holidays = holidays.append([[0]], ignore_index=True)
        
    holidays.columns = ['Holiday']
    holidays.to_csv(root_path+fname[:-4]+'_holiday.csv', sep=',', header=True, index=False)    

def avg_morbility_ontario(myCls, root_path, fname):
    df1 = myCls.load_data(root_path, '2020_CA_Region_Mobility_Report.csv', sep=',', header=0, index_col=None, mode='READ')
    provinces = ['Alberta', 'British Columbia', 'Manitoba', 'New Brunswick', 'Newfoundland and Labrador', 'Nova Scotia', 'Ontario', 'Prince Edward Island', 'Quebec', 'Saskatchewan']
    for province in provinces:
        df2 = df1.query('sub_region_1 == @province') 
        df3 = myCls.load_data(root_path, fname, sep=',', header=0, index_col=None, mode='READ')
        morbility_data = pd.DataFrame()
        for i in range(len(df3)):
            search_date = df3.values[i,0]
            res = df2.query('date == @search_date')  # ALWAYS CHANGE 'DATE' TO SHORT-FORMAT ON .csv FILE
            divisor = len(res)
            if (divisor > 0):
                col1 = res['retail_and_recreation_percent_change_from_baseline'].sum(axis=0, skipna=True)
                col2 = res['grocery_and_pharmacy_percent_change_from_baseline'].sum(axis=0, skipna=True)
                col3 = res['parks_percent_change_from_baseline'].sum(axis=0, skipna=True)
                col4 = res['transit_stations_percent_change_from_baseline'].sum(axis=0, skipna=True)
                col5 = res['workplaces_percent_change_from_baseline'].sum(axis=0, skipna=True)
                col6 = res['residential_percent_change_from_baseline'].sum(axis=0, skipna=True)            
                morbility_data = morbility_data.append([[search_date, round(col1/divisor, 3), round(col2/divisor, 3), round(col3/divisor, 3), round(col4/divisor, 3), round(col5/divisor, 3), round(col6/divisor, 3)]], ignore_index=True)
               
        morbility_data.columns = ['date', 'retail_and_recreation_change', 'grocery_and_pharmacy_change', 'parks_change', 'transit_stations_change', 'workplaces_change', 'residential_change']
        morbility_data.to_csv(root_path+province+'_morbility_data.csv', sep=',', header=True, index=False)   
        
def case_age_gender_computer(myCls, root_path, fname):
    ### Determine Charset encoding of .csv file (USE ORIGINAL FILE AS DOWNLOADED) ###
    import chardet
    local_file = root_path + 'COVID-19_Case_Details_Age-Group_Canada.csv'
    with open(local_file, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(100000))
    print('The charset encoding is: ', result)
    ### Determine Charset encoding of .csv file (USE ORIGINAL FILE AS DOWNLOADED) ###
    
    df1 = myCls.load_data(root_path, 'COVID-19_Case_Details_Age-Group_Canada.csv', sep=',', header=0, index_col=0, mode='READ')
    provinces = ['Alberta', 'British Columbia', 'Manitoba', 'New Brunswick', 'NL', 'Nova Scotia', 'Ontario', 'PEI', 'Quebec', 'Saskatchewan']
    for prvnce in provinces:
        df2 = df1.query('province == @prvnce') 
        df3 = myCls.load_data(root_path, fname, sep=',', header=0, index_col=None, mode='READ')
        age_gender_data = pd.DataFrame()
        for i in range(len(df3)):
            search_date = df3.values[i,0]
            res_df = df2.query('date_reported == @search_date')  # ALWAYS CHANGE 'DATE' TO SHORT-FORMAT ON .csv FILE
            male_0_34 = 0
            male_35_69 = 0
            male_70_above = 0
            female_0_34 = 0
            female_35_69 = 0
            female_70_above = 0
            generic_bias = 0
            if (len(res_df) > 0):
                for j in range(len(res_df)):
                    if (((res_df.values[j,3] == '<20') and (res_df.values[j,4] == 'Male')) or ((res_df.values[j,3] == '20-29') and (res_df.values[j,4] == 'Male')) or ((res_df.values[j,3] == '30-39') and (res_df.values[j,4] == 'Male'))):
                        male_0_34 = male_0_34 + 1
                    elif (((res_df.values[j,3] == '40-49') and (res_df.values[j,4] == 'Male')) or ((res_df.values[j,3] == '50-59') and (res_df.values[j,4] == 'Male')) or ((res_df.values[j,3] == '60-69') and (res_df.values[j,4] == 'Male'))):
                        male_35_69 = male_35_69 + 1
                    elif (((res_df.values[j,3] == '70-79') and (res_df.values[j,4] == 'Male')) or ((res_df.values[j,3] == '80+') and (res_df.values[j,4] == 'Male')) or ((res_df.values[j,3] == 'Not Reported') and (res_df.values[j,4] == 'Male'))):
                        male_70_above = male_70_above + 1
                    elif (((res_df.values[j,3] == '<20') and (res_df.values[j,4] == 'Female')) or ((res_df.values[j,3] == '20-29') and (res_df.values[j,4] == 'Female')) or ((res_df.values[j,3] == '30-39') and (res_df.values[j,4] == 'Female'))):
                        female_0_34 = female_0_34 + 1
                    elif (((res_df.values[j,3] == '40-49') and (res_df.values[j,4] == 'Female')) or ((res_df.values[j,3] == '50-59') and (res_df.values[j,4] == 'Female')) or ((res_df.values[j,3] == '60-69') and (res_df.values[j,4] == 'Female'))):
                        female_35_69 = female_35_69 + 1
                    elif (((res_df.values[j,3] == '70-79') and (res_df.values[j,4] == 'Female')) or ((res_df.values[j,3] == '80+') and (res_df.values[j,4] == 'Female')) or ((res_df.values[j,3] == 'Not Reported') and (res_df.values[j,4] == 'Female'))):
                        female_70_above = female_70_above + 1                    
                    elif (res_df.values[j,4] == 'Not Reported'):
                        generic_bias = generic_bias + 1
                
                # 'generic_bias' resolution & file update
                bias = generic_bias / 6
                if ((generic_bias % 6) == 1):
                    age_gender_data = age_gender_data.append([[search_date, male_0_34+math.floor(bias), male_35_69+math.floor(bias), male_70_above+math.ceil(bias), female_0_34+math.floor(bias), female_35_69+math.floor(bias), female_70_above+math.floor(bias)]], ignore_index=True)
                elif ((generic_bias % 6) == 2):
                    age_gender_data = age_gender_data.append([[search_date, male_0_34+math.floor(bias), male_35_69+math.floor(bias), male_70_above+math.ceil(bias), female_0_34+math.floor(bias), female_35_69+math.floor(bias), female_70_above+math.ceil(bias)]], ignore_index=True)
                elif ((generic_bias % 6) == 3):
                    age_gender_data = age_gender_data.append([[search_date, male_0_34+math.floor(bias), male_35_69+math.ceil(bias), male_70_above+math.ceil(bias), female_0_34+math.floor(bias), female_35_69+math.floor(bias), female_70_above+math.ceil(bias)]], ignore_index=True)
                elif ((generic_bias % 6) == 4):
                    age_gender_data = age_gender_data.append([[search_date, male_0_34+math.floor(bias), male_35_69+math.ceil(bias), male_70_above+math.ceil(bias), female_0_34+math.floor(bias), female_35_69+math.ceil(bias), female_70_above+math.ceil(bias)]], ignore_index=True)
                elif ((generic_bias % 6) == 5):
                    age_gender_data = age_gender_data.append([[search_date, male_0_34+math.ceil(bias), male_35_69+math.ceil(bias), male_70_above+math.ceil(bias), female_0_34+math.floor(bias), female_35_69+math.ceil(bias), female_70_above+math.ceil(bias)]], ignore_index=True)
                elif ((generic_bias % 6) == 0):
                    age_gender_data = age_gender_data.append([[search_date, male_0_34+math.ceil(bias), male_35_69+math.ceil(bias), male_70_above+math.ceil(bias), female_0_34+math.ceil(bias), female_35_69+math.ceil(bias), female_70_above+math.ceil(bias)]], ignore_index=True)
            else:
                age_gender_data = age_gender_data.append([[search_date, male_0_34, male_35_69, male_70_above, female_0_34, female_35_69, female_70_above]], ignore_index=True)
        age_gender_data.columns = ['date', 'COV_M_0_34', 'COV_M_35_69', 'COV_M_70_above', 'COV_F_0_34', 'COV_F_35_69', 'COV_F_70_above']
        age_gender_data.to_csv(root_path+prvnce+'_age_gender_data.csv', sep=',', header=True, index=False)
        
def feat_extractor(dframe, label='train'):    
    TotPopln = dframe.loc[:,'0 - 34 (M)'] + dframe.loc[:,'35 - 69 (M)'] + dframe.loc[:,'70 - Above (M)'] + dframe.loc[:,'0 - 34 (F)'] + dframe.loc[:,'35 - 69 (F)'] + dframe.loc[:,'70 - Above (F)']
    dframe['mobility_index'] = dframe.loc[:,'retail_and_recreation_change'] + dframe.loc[:,'grocery_and_pharmacy_change'] + dframe.loc[:,'parks_change'] + dframe.loc[:,'transit_stations_change'] + dframe.loc[:,'workplaces_change'] + dframe.loc[:,'residential_change']
    dframe['mobility_totpop'] = dframe.loc[:,'mobility_index'].div(TotPopln, axis=0)
    dframe['infect_mobility'] = dframe.loc[:,'Virus Reprod Index'].mul(dframe['mobility_index'], axis=0)
    dframe['wave_climate'] = dframe.loc[:,'Wave'].div(dframe['Climate'], axis=0)
    dframe['infect_totpop_ratio'] = dframe.loc[:,'Virus Reprod Index'].div(TotPopln, axis=0)
    dframe['infect_travel_ratio'] = dframe.loc[:,'Virus Reprod Index'].div(dframe['Return Travellers'], axis=0)
    dframe['infect_M_0_34'] = dframe.loc[:,'Virus Reprod Index'].div(dframe['0 - 34 (M)'], axis=0)
    dframe['infect_M_35_69'] = dframe.loc[:,'Virus Reprod Index'].div(dframe['35 - 69 (M)'], axis=0)
    dframe['infect_M_70_above'] = dframe.loc[:,'Virus Reprod Index'].div(dframe['70 - Above (M)'], axis=0)
    dframe['infect_F_0_34'] = dframe.loc[:,'Virus Reprod Index'].div(dframe['0 - 34 (F)'], axis=0)
    dframe['infect_F_35_69'] = dframe.loc[:,'Virus Reprod Index'].div(dframe['35 - 69 (F)'], axis=0)
    dframe['infect_F_70_above'] = dframe.loc[:,'Virus Reprod Index'].div(dframe['70 - Above (F)'], axis=0)
    dframe['immunity_ratio'] = dframe.loc[:,'Cumm Vaccine'].div(TotPopln, axis=0)
    dframe['travel_totpop'] = dframe.loc[:,'Return Travellers'].div(TotPopln, axis=0)
    dframe['travel_land'] = dframe.loc[:,'Return Travellers'].div(dframe['Dry Land'], axis=0)
    dframe['pop_density'] = TotPopln.div(dframe['Dry Land'], axis=0)
    dframe['chc_density'] = dframe.loc[:,'CHCentres'].div(dframe['Dry Land'], axis=0)
    print("Shape of 'supplemented' ", label, " data: ", dframe.shape)
    
def ppe_pred_dset(myCls, root_path, fname):
    df1 = myCls.load_data(root_path, fname, sep=',', header=0, index_col=None, mode='READ')
    ppe_pred = pd.DataFrame()
    for i in range(len(df1)):
        # ppe_pred
        CHCentres = 153
        HealthPersons = 7253 + 8996 + 519 + 34094 + 160137 + 3446  # MedLabTech(7253) + Parameds(8996) + PhysicianAssts(519) + Physicians(34094) + Nurses(160137) + RespiratoryTherapists(3446)
        Cap_HealthPersons = 0.85
        hosptlztn = df1.values[i,2]
        hosptlztn_ratio = hosptlztn/CHCentres
        if (hosptlztn_ratio >= 1):  # There is at least '1' COVID19 hospitalization-case per CHCentre
            ppe_qty = Cap_HealthPersons * HealthPersons * 1
        else:
            ppe_qty = Cap_HealthPersons * HealthPersons * hosptlztn_ratio        
        ppe_pred = ppe_pred.append([[ppe_qty]], ignore_index=True)
        
    y_data = pd.concat([pd.DataFrame(df1.values[:,0]), pd.DataFrame(df1.values[:,1]), pd.DataFrame(df1.values[:,2]), ppe_pred, pd.DataFrame(df1.values[:,3]), pd.DataFrame(df1.values[:,4])], axis='columns')
    y_data.columns = ['Reported Date', 'Infected Positive (I)', 'Hospitalized (incls. ICU)', 'PPE Qty', 'Recovered (R)', 'Deaths (D)']
    y_data.to_csv(root_path+'covidtesting_ontario_hospitalized.csv', sep=',', header=True, index=False)
    
def covid19_prevalence(myCls, root_path, fname):
    df1 = myCls.load_data(root_path, fname, sep=',', header=0, index_col=None, mode='READ')
    
    # Plot performance of the Network-Model fitting on the dataset
    plt.figure(figsize=(32,15))  # Using "plt.figure()" to tweak the resultant graph plot  
    
    plt.subplot(1, 2, 1)  # subplot(rows, cols, active_ROW-COL_in_subplotSpace)
    plt.grid()
    plt.plot(df1.loc[:,'COV_M_0_34'], 'k-', marker='o', label='0 - 34 (Male Age Group)')
    plt.plot(df1.loc[:,'COV_M_35_69'], 'b--', marker='*', label='35 - 69 (Male Age Group)')
    plt.plot(df1.loc[:,'COV_M_70_above'], 'g:', marker='D', label='70 - Above (Male Age Group)')
    plt.xlabel('Days (Jan. 25, 2020 - Jan. 20, 2021)')
    plt.ylabel('Number of Infection/Positive Cases')
    plt.legend(loc='best')
    plt.title('SARS-CoV-2 Infection Prevalence in Ontario (Male Age Group)')
    
    plt.subplot(1, 2, 2)  # subplot(rows, cols, active_ROW-COL_in_subplotSpace)
    plt.grid()
    plt.plot(df1.loc[:,'COV_F_0_34'], 'k-', marker='o', label='0 - 34 (Female Age Group)')
    plt.plot(df1.loc[:,'COV_F_35_69'], 'b--', marker='*', label='35 - 69 (Female Age Group)')
    plt.plot(df1.loc[:,'COV_F_70_above'], 'g:', marker='D', label='70 - Above (Female Age Group)')
    plt.xlabel('Days (Jan. 25, 2020 - Jan. 20, 2021)')
    plt.ylabel('Number of Infection/Positive Cases')
    plt.legend(loc='best')
    plt.title('SARS-CoV-2 Infection Prevalence in Ontario (Female Age Group)')
    
    plt.savefig(root_path + 'plots_and_data/COVID-19_Prevalence_Males_and_Females.png')
    plt.show()    
    
def ppe_pred(data):
    data = list(data)
    ppe_pred = list()
    for i in range(len(data)):
        # ppe_pred
        CHCentres = 153
        HealthPersons = 7253 + 8996 + 519 + 34094 + 160137 + 3446  # MedLabTech(7253) + Parameds(8996) + PhysicianAssts(519) + Physicians(34094) + Nurses(160137) + RespiratoryTherapists(3446)
        Cap_HealthPersons = 0.85
        hosptlztn = data[i]
        hosptlztn_ratio = hosptlztn/CHCentres
        if (hosptlztn_ratio >= 1):  # There is at least '1' COVID19 hospitalization-case per CHCentre
            ppe_qty = Cap_HealthPersons * HealthPersons * 1
        else:
            ppe_qty = Cap_HealthPersons * HealthPersons * hosptlztn_ratio        
        ppe_pred.append(np.rint(ppe_qty).astype(np.int32))
        
    return ppe_pred

def percent_comp(data):
    data = list(data)
    cap = max(data)
    new_list = list()
    for i in range(len(data)):
        percent = (data[i]/cap) * 100
        percent = np.rint(percent).astype(np.int32)
        new_list.append(str(percent)+'%')
        
    return new_list
    
#################################################################### Program Flow ####################################################################

def main_prog_flow(myCls, args):
    # ".iloc[]" returns a Pandas DATAFRAME
    # ".values[]" returns a NUMPY Array wrt dataframes
    #cumm_2_reg_conv(myCls, args.root_path, 'covidtesting_ontario.csv')
    #wkend_holiday(myCls, args.root_path, 'covidtesting_ontario.csv')
    #avg_morbility_ontario(myCls, args.root_path, 'covidtesting_ontario.csv')
    #case_age_gender_computer(myCls, args.root_path, 'covidtesting_ontario.csv')
    
    # MISSING DATA: 'SimpleImputer' or 'IterativeImputer'
    # STANDARDIZATION (Column-wise): Scale each/individual features (columns) to have values within a fixed range, usually [0,1]
    # NON-LINEAR TRANSFORMATION (Column-wise): 'QuantileTransformer()' transforms features (columns) containing 'skewed/congested' or 'highly-spread' data into a standard normal/probability distribution 
    # NORMALIZATION (Row-wise): Scale each/individual samples (rows) to usually a unit-norm (square of all columns in the row == 1)
    
    ### Generic hyperparameters
    model = "ML"  # TL/MTL | ML
    train_frac = 0.85
    test_frac = 1 - train_frac
    
    ### Load and Aggregate datasets                
    if (not os.path.isfile(args.root_path+'covidtesting_train_data.csv')) or (not os.path.isfile(args.root_path+'covidtesting_test_data.csv')):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        cols_aft_drop = ['Infected Positive (I)', 'Hospitalized (incls. ICU)', 'Recovered (R)', 'Deaths (D)', 'Cumm Recovered', 'Cumm Deaths', 'Region', 'Wave', 'Cumm Vaccine', 'Virus Reprod Index', 'Lockdown', 'Travel Restrict', 'Province FaceCover', 'Holiday', 'Climate', 'retail_and_recreation_change', 'grocery_and_pharmacy_change', 'parks_change', 'transit_stations_change', 'workplaces_change', 'residential_change', 'Return Travellers', 'Employ Rate', 'Unemploy Rate', 'Labor Popln', 'Dry Land', 'CHCentres', '0 - 34 (M)', '35 - 69 (M)', '70 - Above (M)', '0 - 34 (F)', '35 - 69 (F)', '70 - Above (F)']
        dset_suffix = ['alberta', 'british_columbia', 'manitoba', 'new_brunswick', 'ontario', 'quebec', 'saskatchewan']
        for suffix in dset_suffix:
            df = myCls.load_data(args.root_path, 'covidtesting_'+suffix+'.csv', sep=',', header=0, index_col=0, mode='READ')
            df.drop(['Labor Popln (x1000)', 'Males', 'Avg Age (M)', 'Females', 'Avg Age (F)', 'Life Expectancy', 'COV_M_0_34', 'COV_M_35_69', 'COV_M_70_above', 'COV_F_0_34', 'COV_F_35_69', 'COV_F_70_above'], axis='columns', inplace=True)
            # MISSING DATA: 'SimpleImputer' or 'IterativeImputer'
            missing_data_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputed_data = missing_data_imputer.fit_transform(df.values[:,:])
            ### 'ontario'-VALIDATION SAMPLE ###
            if (suffix == 'ontario'):
                test_val = np.rint(test_frac * len(imputed_data)).astype(np.int32)
                ontario_dset_idx = [a for a in range(len(imputed_data))]
                unique_subset = sample(ontario_dset_idx, test_val)  # Selects a random/shuffled subset without replacement (UNIQUE SUBSET)
                temp = pd.DataFrame(imputed_data[unique_subset,:])
                test_data = test_data.append(temp, ignore_index=True)
                test_data.columns = cols_aft_drop
                test_data.to_csv(args.root_path+'covidtesting_test_data.csv', sep=',', header=True, index=False)
                imputed_data = pd.DataFrame(imputed_data)
                imputed_data.drop(unique_subset, axis='index', inplace=True)
            ### 'ontario'-VALIDATION SAMPLE ###
            train_data = train_data.append(pd.DataFrame(imputed_data), ignore_index=True)
        train_data.columns = cols_aft_drop
        train_data.to_csv(args.root_path+'covidtesting_train_data.csv', sep=',', header=True, index=False)
    else:
        train_data = myCls.load_data(args.root_path, 'covidtesting_train_data.csv', sep=',', header=0, index_col=None, mode='READ')
        test_data = myCls.load_data(args.root_path, 'covidtesting_test_data.csv', sep=',', header=0, index_col=None, mode='READ')
        
    train_data = train_data.sample(frac=1, random_state=42, axis='index')  # Shuffle 'train' dataset
    test_data = test_data.sample(frac=1, random_state=42, axis='index')  # Shuffle 'test' dataset
    print("Shape of 'original' train data: ", train_data.shape)
    print("Shape of 'original' test data: ", test_data.shape)

    ### Feature Extraction: Computation of additional/supplementary features
    feat_extractor(train_data, label='train')
    feat_extractor(test_data, label='test')

    ### Data Visualizations
    print(train_data.head(n=5))  # First 5 rows of dataset
    print(train_data.info(verbose=True))  # Basic information related to dataset
    #print(train_data.tail(n=10))  # Last 10 rows of dataset
    #print(train_data.describe())  # Describe basic statistical information wrt dataset
    # Show Histogram-plot of dataset (bin == buckets/classes)
    disp_cols = ['Infected Positive (I)', 'Hospitalized (incls. ICU)', 'Recovered (R)', 'Deaths (D)', 'Cumm Recovered', 'Cumm Deaths', 'Region', 'Wave', 'Cumm Vaccine', 'Virus Reprod Index', 'Lockdown', 'Travel Restrict', 'Province FaceCover', 'Holiday', 'Climate', 'retail_and_recreation_change', 'grocery_and_pharmacy_change', 'parks_change', 'transit_stations_change', 'workplaces_change', 'residential_change', 'Return Travellers', 'Employ Rate', 'Unemploy Rate', 'Labor Popln']
    data_hist = train_data.hist(column=disp_cols, xlabelsize=20, ylabelsize=20, figsize=(22,21), bins=50)
    print(data_hist)
    
    ### Feature Selection:
    train_X = train_data.values[:,7:]
    train_y = train_data.values[:,:4]
    test_X = test_data.values[:,7:]
    test_y = test_data.values[:,:4]
    data_directory = 'plots_and_data/covidtesting'
    fname = args.root_path + data_directory
    X_extract_cols = ['Wave', 'Cumm Vaccine', 'Virus Reprod Index', 'Lockdown', 'Travel Restrict', 'Province FaceCover', 'Holiday', 'Climate', 'retail_and_recreation_change', 'grocery_and_pharmacy_change', 'parks_change', 'transit_stations_change', 'workplaces_change', 'residential_change', 'Return Travellers', 'Employ Rate', 'Unemploy Rate', 'Labor Popln', 'Dry Land', 'CHCentres', '0 - 34 (M)', '35 - 69 (M)', '70 - Above (M)', '0 - 34 (F)', '35 - 69 (F)', '70 - Above (F)', 'mobility_index', 'mobility_totpop', 'infect_mobility', 'wave_climate', 'infect_totpop_ratio', 'infect_travel_ratio', 'infect_M_0_34', 'infect_M_35_69', 'infect_M_70_above', 'infect_F_0_34', 'infect_F_35_69', 'infect_F_70_above', 'immunity_ratio', 'travel_totpop', 'travel_land', 'pop_density', 'chc_density']
    #X_extract_cols = ['0: Wave', '1: Cumm Vaccine', '2: Virus Reprod Index', '3: Lockdown', '4: Travel Restrict', '5: Province FaceCover', '6: Holiday', '7: Climate', '8: retail_and_recreation_change', '9: grocery_and_pharmacy_change', '10: parks_change', '11: transit_stations_change', '12: workplaces_change', '13: residential_change', '14: Return Travellers', '15: Employ Rate', '16: Unemploy Rate', '17: Labor Popln', '18: Dry Land', '19: CHCentres', '20: 0 - 34 (M)', '21: 35 - 69 (M)', '22: 70 - Above (M)', '23: 0 - 34 (F)', '24: 35 - 69 (F)', '25: 70 - Above (F)', '26: mobility_index', '27: mobility_totpop', '28: infect_mobility', '29: wave_climate', '30: infect_totpop_ratio', '31: infect_travel_ratio', '32: infect_M_0_34', '33: infect_M_35_69', '34: infect_M_70_above', '35: infect_F_0_34', '36: infect_F_35_69', '37: infect_F_70_above', '38: immunity_ratio', '39: travel_totpop', '40: travel_land', '41: pop_density', '42: chc_density']
    y_labels = ['Infected +ve (I) in Ontario', 'Hospitalized (H) in Ontario', 'Recovered (R) in Ontario', 'Deaths (D) in Ontario']
    
    # Compute Correlation-Coefficients matrix
    if model == 'TL/MTL':
        myCls.correlation_coefs(train_data, 'Infected Positive (I)', fname)
    
    # 'ExtraTreesRegressor' Feature-Selection strategy
    top_feats = ExtraTreesRegressor()
    top_feats = top_feats.fit(train_X, train_y[:,0])    
    feats_score = pd.Series(top_feats.feature_importances_)
    feats_percent = percent_comp(top_feats.feature_importances_)
    feats_percent = pd.Series(feats_percent)
    cols = pd.Series(X_extract_cols)
    feats_rank = pd.concat([cols, feats_score, feats_percent], axis='columns')
    feats_rank.columns = ['Feature/Variable', 'Score', 'Percentage']
    print(feats_rank.nlargest(12, 'Score'))  # Sort pandas dataframe by 'Score    
    feats_score.index = X_extract_cols
    feats_score.nlargest(12).plot(kind='barh', grid=True, figsize=(33,15))
    plt.xlabel('Relevance Score')
    plt.title('Feature Selection (ExtraTreesRegressor Top-k Features): ' + y_labels[0])
    plt.savefig(fname[:-12] + y_labels[0] + '_ExtraTreesRegr_Features.png')
    plt.show()
    
    # 'SelectKBest' Feature-Selection strategy
    for p in range(train_y.shape[1]):
        top_22_feats = SelectKBest(score_func=f_regression, k=12)
        top_22_feats = top_22_feats.fit(train_X, train_y[:,p])
        relv_score = pd.Series(top_22_feats.scores_)
        relv_percent = percent_comp(top_22_feats.scores_)
        relv_percent = pd.Series(relv_percent)
        cols = pd.Series(X_extract_cols)
        feat_ranking = pd.concat([cols, relv_score, relv_percent], axis='columns')
        feat_ranking.columns = ['Feature/Variable', 'Score', 'Percentage']
        print(feat_ranking.nlargest(12, 'Score'))  # Sort pandas dataframe by 'Score'  
        relv_score.index = X_extract_cols 
        relv_score.nlargest(12).plot(kind='barh', grid=True, figsize=(33,15))
        plt.xlabel('Relevance Score')
        plt.title('Feature Selection (SelectKBest Top-k Features): ' + y_labels[p])
        plt.savefig(fname[:-12] + y_labels[p] + '_SelectKBest_Features.png')
        plt.show()
    
    ### Feature Scaling:
    #X_train = top_22_feats.transform(train_X)
    X_train = train_X[:,[0, 21, 17, 22, 24, 25, 20, 23, 19, 1, 18, 13, 3]]
    y_train = train_y
    #X_test = top_22_feats.transform(test_X)
    X_test = test_X[:,[0, 21, 17, 22, 24, 25, 20, 23, 19, 1, 18, 13, 3]]
    y_test = test_y
    # NON-LINEAR TRANSFORMATION (Column-wise) & STANDARDIZATION (Column-wise): 'QuantileTransformer()' transforms features (columns) containing 'skewed/congested' or 'highly-spread' data into a standard normal/probability distribution 
    quantile_trans_standzatn = QuantileTransformer(output_distribution='uniform', random_state=42)
    quantile_trans_standzatn = quantile_trans_standzatn.fit(X_train)
    standized_X_train = quantile_trans_standzatn.transform(X_train)
    standized_X_test = quantile_trans_standzatn.transform(X_test)    
    minmax_y_scaler = MinMaxScaler(feature_range=(0, 1))
    minmax_y_scaler = minmax_y_scaler.fit(y_train)
    standized_y_train = minmax_y_scaler.transform(y_train)
    standized_y_test = minmax_y_scaler.transform(y_test)
    
    # NORMALIZATION (Row-wise): Scale each/individual samples (rows) to usually a unit-norm (square of all columns in the row == 1)
    normaliztn = Normalizer(norm='l2')
    normaliztn = normaliztn.fit(standized_X_train)
    normalized_X_train = normaliztn.transform(standized_X_train)
    normalized_X_test = normaliztn.transform(standized_X_test)
    
    ### Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strategeies
    strat_X_train = normalized_X_train  ##################################################### standized_X_train # normalized_X_train #####################################################
    strat_y_train = standized_y_train
    strat_X_test = normalized_X_test  ##################################################### standized_X_test # normalized_X_test #####################################################
    strat_y_test = standized_y_test
    print("\nTraining model using {:.2f}% data...".format(train_frac * 100))
    print("Shape of 'strat_X_train': %s;  Shape of 'strat_y_train': %s" % (strat_X_train.shape, strat_y_train.shape))
    print("Shape of 'strat_X_test': %s;  Shape of 'strat_y_test': %s" % (strat_X_test.shape, strat_y_test.shape))
    ### Preserve ratio/percentage of samples per class using efficent data-splitting && data-resampling strategeies

    ### Fit ML/DL model to training data
    mlearning_model = "KNN"  # KNN, GradBoost, RandFrst, ExtraTree, DeciTree, etc
    start_time = time.time()  # START: Training Time Tracker    
    if (mlearning_model == "LinearRegress"):
        model = MultiOutputRegressor(LinearRegression())
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "LogisticRegress"):
        model = MultiOutputRegressor(LogisticRegression(solver='lbfgs', random_state=42))
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "SuppVec"):
        model = MultiOutputRegressor(SVR(epsilon=0.1))
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "StochastGrad"):
        model = MultiOutputRegressor(SGDRegressor(learning_rate='invscaling', eta0=0.01, random_state=42))
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "N_Bayes_Ridge"):
        model = MultiOutputRegressor(BayesianRidge())
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "N_Bayes_ARD"):
        model = MultiOutputRegressor(ARDRegression())
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "GradBoost"):
        model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "DeciTree"):    
        model = DecisionTreeRegressor(random_state=42)
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "RandFrst"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "ExtraTree"):
        model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "KNN"):
        model = KNeighborsRegressor(n_neighbors=6, weights='distance')
        model.fit(strat_X_train, strat_y_train)
    elif (mlearning_model == "DL_MLP"):
        model = MLPRegressor(hidden_layer_sizes=(250, 300, 500, 500, 500, 500, 500, 500, 100, strat_y_train.shape[1]), activation='relu', solver='lbfgs', learning_rate='constant', learning_rate_init=0.001, random_state=42)
        model.fit(strat_X_train, strat_y_train)
    end_time = time.time()  # STOP: Training Time Tracker
    print("\nTraining Time: ", end_time - start_time, "seconds")  # PRINT: Training Time Tracker
    
    ### TRAINING: Evaluate model's performance (OVERFITTING = Train MSE < Test MSE || UNDERFITTING = Train MSE > Test MSE)
    pred_y_train = model.predict(strat_X_train)
    train_mse = mean_squared_error(strat_y_train, pred_y_train)
    train_rmse = math.sqrt(train_mse)
    print('TRAINING MSE//RMSE for Prediction(s): %.3f//%.3f' % (train_mse, train_rmse))
    
    ### Forecasting/Prediction: Make Predictions for 'y-component' on the validation/test dataset
    pred_y_test = model.predict(strat_X_test)    
    reversed_y_test = minmax_y_scaler.inverse_transform(pred_y_test)  # Invert Scaling: For 'y-component' only
    reversed_y_test = np.rint(reversed_y_test).astype(np.int32)
    
    ### GENERALIZATION: Evaluate model's performance (OVERFITTING = Train MSE < Test MSE || UNDERFITTING = Train MSE > Test MSE)
    raw_y_test = test_y    
    mse, rmse, mae, expvar, r2 = list(), list(), list(), list(), list()
    for x in range(raw_y_test.shape[1]):        
        temp_mse = mean_squared_error(raw_y_test[:,x], reversed_y_test[:,x])  # BEST = 0.0, WORST = +ve
        temp_rmse = math.sqrt(temp_mse)  # BEST = 0.0, WORST = +ve
        temp_mae = mean_absolute_error(raw_y_test[:,x], reversed_y_test[:,x])  # BEST = 0.0, WORST = +ve
        temp_expvar = explained_variance_score(raw_y_test[:,x], reversed_y_test[:,x])  # BEST = 1.0, WORST = 0.0
        temp_r2 = r2_score(raw_y_test[:,x], reversed_y_test[:,x])  # BEST = 1.0, WORST = -ve
        mse.append(temp_mse)
        rmse.append(temp_rmse)
        mae.append(temp_mae)
        expvar.append(temp_expvar)
        r2.append(temp_r2)
        print('\n\nVALIDATION MSE//RMSE for Forecast/Prediction: %.3f//%.3f' % (temp_mse, temp_rmse))
        print('VALIDATION MAE//EXP_VAR for Forecast/Prediction: %.3f//%.3f' % (temp_mae, temp_expvar))
        print('VALIDATION R2 for Forecast/Prediction: %.3f' % (temp_r2))   
        
        # Plot line graphs of Groundtruths Vs. Predictions
        myCls.graph_eval_2(raw_y_test[:,x], reversed_y_test[:,x], fname, leg_label=y_labels[x], title='COVID-19')
        
        # PPE-Kit Prediction
        if (x == 1):  # Prediction data == 'Hospitalized (H) in Ontario'
            ppe_kits = ppe_pred(reversed_y_test[:,x])
            hospt_ppe = pd.concat([pd.Series(reversed_y_test[:,x]), pd.Series(ppe_kits)], axis='columns')
            hospt_ppe.columns = ['Hospitalized (H) in Ontario', 'PPE-Kit Demand in Ontario']
            print(hospt_ppe)
            
    # Plot of COVID-19 Prevalence among Age Groups (Males & Females)
    covid19_prevalence(myCls, args.root_path, 'covidtesting_ontario.csv') 
    
    # Summarize results
    cumm_err_results = pd.concat([pd.Series(mse), pd.Series(rmse), pd.Series(mae), pd.Series(expvar), pd.Series(r2)], axis=1, ignore_index=True)  # Returns dataframe
    cumm_err_results.columns = ['MSE', 'RMSE', 'MAE', 'EXP_VAR', 'R2']
    print(cumm_err_results.describe())
    cumm_err_results.boxplot()
    plt.title('Box-Plot of y_data_Predictions Vs. Regression_Metrics')
    plt.savefig(fname + '_boxplot_ML_predict_performance.png')
    plt.show()
        
def loop_prog_flow(myCls, args):
    #graph_data = ["CiteSeer", "Cora", "DBLP", "Facebook-Page2Page", "PubMed-Diabetes", "Wiki", "Zachary-Karate"]
    edge_list = os.listdir(args.root_path)

    for i in range(len(edge_list)):
        main_prog_flow(myCls, args, edge_list[i])
    
def entry_point():    
    pd.set_option('display.max_columns', None)  # Force Pandas() to display any number of columns
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # Force Numpy() to display any number of columns
    
    # Parse arguments from command-line
    args = args_parse_cmd()
    
    # Load & instantiate generic dependency (class file)
    REMOTE_URL = "https://snap.stanford.edu/data/gemsec_deezer_dataset.tar.gz"
    ROOT_PATH = args.root_path
    depedencyClass = Starter(REMOTE_URL, ROOT_PATH)
    
    # Parse program run-mode
    if args.run_mode == 'single':
        main_prog_flow(depedencyClass, args)
    elif args.run_mode == 'all':
        loop_prog_flow(depedencyClass, args)
        
if __name__ == "__main__":
    entry_point()
