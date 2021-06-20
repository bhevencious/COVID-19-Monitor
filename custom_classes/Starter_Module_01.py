# Enforce CPU Usage
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Seed the Random-Number-Generator in a bid to get 'Reproducible Results'
import tensorflow as tf
from numpy.random import seed
seed(1)
tf.compat.v1.set_random_seed(3)

# load required modules
from datetime import datetime
from six.moves import urllib
import pandas as pd
import numpy as np
import seaborn as sbn
import json, csv, math, tarfile, zipfile, time
import matplotlib.pyplot as plt
size_01 = 14
size_02 = 24
plt.rc('font', size=size_02)  # controls default text sizes
plt.rc('axes', titlesize=size_02)  # fontsize of the axes title
plt.rc('axes', labelsize=size_02)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=size_02)  # fontsize of the tick labels
plt.rc('ytick', labelsize=size_02)  # fontsize of the tick labels
plt.rc('legend', fontsize=size_02)  # legend fontsize
plt.rc('figure', titlesize=size_02)  # fontsize of the figure title

'''
# Enforce GPU Usage
with tf.device('/gpu:1'):
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8, allow_soft_placement=True, device_count = {'CPU': 1, 'GPU': 1})
    session = tf.compat.v1.Session(config=config)
    K.set_session(session)
'''


class Starter():
    def __init__(self, remote_url, local_path):
        ''' Constructor for this class '''
        self.remote_url = remote_url
        self.local_path = local_path

    # CUSTOM: Format date in "shampoo_sales" dataset (used in "load_data()" function)
    def custom_date_parser(self, raw_date):
        return datetime.strptime(raw_date, '%Y %m %d %H')
    
    # Extract data from remote directory to LOCAL directory
    def fetch_data(self, remote_url, local_path, file_name):
        if not os.path.isdir(local_path):
            os.makedirs(local_path)
        local_file = os.path.join(local_path, file_name)
        if not os.path.isfile(local_file):    
            urllib.request.urlretrieve(remote_url, local_file)
            if (file_name[-7:] == ".tar.gz"):
                file_tgz = tarfile.open(local_file)
                file_tgz.extractall(path=local_path)
                file_tgz.close()
            elif (file_name[-4:] == ".zip"):
                file_zip = zipfile.ZipFile(local_file)
                file_zip.extractall(path=local_path)
                file_zip.close()
            elif (file_name[-4:] == ".rar"):
                file_rar = rarfile.RarFile(local_file)
                file_rar.extractall(path=local_path)
                file_rar.close()
    
    # Load data from LOCAL directory after REMOTE extraction    
    def load_data(self, local_path, file_name, sep="\s", header=0, index_col=0, mode="EXTRACT"):
        local_file = local_path + file_name
        if (file_name[-5:] == ".xlsx") or (file_name[-4:] == ".xls"):
            if (mode == "EXTRACT"):
                return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0, parse_dates = [['year', 'month', 'day', 'hour']], date_parser=self.custom_date_parser)
            elif (mode == "GRAPH"):
                return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0)
            else:
                return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0)
        elif (file_name[-4:] == ".csv"):
            if (mode == "EXTRACT"):
                return pd.read_csv(local_file, header=header, index_col=index_col, parse_dates = [['year', 'month', 'day', 'hour']], date_parser=self.custom_date_parser)
            elif (mode == "GRAPH"):
                return pd.read_csv(local_file, header=header, index_col=index_col)
            else:
                return pd.read_csv(local_file, header=header, index_col=index_col)        
        else:
            return pd.read_table(local_file, sep=sep, header=header, index_col=index_col, engine='python')
        
    # Performance evaluator graphs
    def graph_eval_1(self, repeats, mdl_history, suffix, mode="DEFAULT", labels="DEFAULT"):
        # Plot performance of the Network-Model fitting on the dataset
        plt.figure(figsize=(22,11))  # Using "plt.figure()" to tweak the resultant graph plot  
        if (mode == "DEFAULT"):
            a1 = mdl_history.history['loss'] ; a2 = mdl_history.history['val_loss']
            b1 = mdl_history.history['acc'] ; b2 = mdl_history.history['val_acc']
        elif (mode == "CUSTOM"):
            a1 = mdl_history[:,0] ; a2 = mdl_history[:,1]
            b1 = mdl_history[:,2] ; b2 = mdl_history[:,3]
            
        # Resolve graph labels
        title = suffix.split('_')[-1]
        if (labels == "DEFAULT"): 
            cost_func = "Categorical CrossEntropy"
            fitness_func = "Categroical Accuracy"
        else:
            cost_func = labels[0]
            fitness_func = labels[1]
            
        plt.subplot(1, 2, 1)  # subplot(rows, cols, active_ROW-COL_in_subplotSpace)
        plt.grid()
        plt.plot(a1, "k-", marker='o', label=cost_func+' (Training)')
        plt.plot(a2, "b--", marker='o', label=cost_func+' (Validation/Test)')
        plt.xlabel('Epochs')
        plt.ylabel(cost_func+' (Cost Function)')
        plt.legend(loc="best")
        plt.title(title + " Dataset")
        
        plt.subplot(1, 2, 2)  # subplot(rows, cols, active_ROW-COL_in_subplotSpace)
        plt.grid()
        plt.plot(b1, "k-", marker='o', label=fitness_func+' (Training)')
        plt.plot(b2, "b--", marker='o', label=fitness_func+' (Validation/Test)')
        plt.xlabel('Epochs')
        plt.ylabel(fitness_func+' (Fitness Function)')
        plt.legend(loc="best")
        plt.title(title + " Dataset")
        
        plt.savefig(self.local_path + "/fitting_learning_" + str(repeats) + "_" + suffix + ".png")
        plt.show()
        
    # Performance evaluator graphs
    def graph_eval_2(self, gtruth, pred, fname, leg_label='y_data', title='COVID-19'):
        # Plot performance of the model fitting on the dataset
        plt.figure(figsize=(21,15))  # Using "plt.figure()" to tweak the resultant graph plot  
                    
        plt.subplot(1, 1, 1)  # subplot(rows, cols, active_ROW-COL_in_subplotSpace)
        plt.grid()
        plt.plot(gtruth, 'k-', marker='o', label='Ground Truth ('+leg_label+')')
        plt.plot(pred, 'b--', marker='o', label='Prediction ('+leg_label+')')
        plt.xlabel('Randomly Sampled Days/Dates (Jan. 2020 - Jan. 2021)')
        plt.ylabel('Count/Frequency of Cases')
        plt.legend(loc='best')
        plt.title(title + ' Data-Plot - ' + leg_label)
        
        plt.savefig(fname + '_' + leg_label.replace(" ", "_") + '_ML_predict_performance.png')
        plt.show()
        
    def correlation_coefs(self, data, y_target, fname):
        # 'data' MUST be of type == 'PANDAS Dataframe'
        plt.figure(figsize=(64,64))  # Using "plt.figure()" to tweak the resultant graph plot
        title = fname.split('/')[-1]
        data_corr = data.corr('kendall')  # Compute correlation coefficients. Linear ('pearson'). Nonlinear ('kendall' or 'spearman')
        
        # Output ONLY highly correlated variables
        corr_target = abs(data_corr[y_target])
        #Selecting highly correlated features
        useful_feats = corr_target[corr_target>=0.49]
        print("\nUseful/Relevant Features:\n-------------------------")
        print(useful_feats)
        
        sbn.heatmap(data_corr, annot=True, linewidths=0.5, fmt='.1g', cmap=plt.cm.Reds)
        plt.xlabel('Features: X = [:,0:-1] and y = [:,-1]')
        plt.ylabel('Features: X = [:,0:-1] and y = [:,-1]')
        plt.title(title + " Dataset")
        plt.savefig(fname + '_corr_map.png')
        plt.show()
        
        pd.plotting.scatter_matrix(data_corr, alpha=0.9, figsize=(64,64), diagonal='kde')
        #plt.xlabel('Features: X = [:,0:-1] and y = [:,-1]')
        #plt.ylabel('Features: X = [:,0:-1] and y = [:,-1]')
        #plt.title(title + " Dataset")
        plt.savefig(fname + '_corr_scatter.png')
        plt.show()