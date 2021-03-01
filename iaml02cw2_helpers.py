#
#  Helper functions for IAML 2020/21 S2 cw2
#
import os
import gzip
import numpy as np
from scipy.io import loadmat

def load_EMNIST_subset(filename='data1.mat'):
    data = loadmat(filename)
    Xtrn_org = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_)
    Xtst_org = data['dataset']['test'][0,0]['images'][0,0][:,:].astype(dtype=np.float_)
    Ytrn_org = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()
    Ytst_org = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()

    return(Xtrn_org, Ytrn_org, Xtst_org, Ytst_org)
    

def load_UniMiB_SHAR_ADL(dir='.'):
    train_idx = loadmat(dir+'/adl_train_idxssubjective_folds.mat')['train_idxs']
    test_idx = loadmat(dir+'/adl_test_idxssubjective_folds.mat')['test_idxs']
    X = np.array(loadmat(dir+'/adl_data.mat')['adl_data'])
    Y = np.array(loadmat(dir+'/adl_labels.mat')['adl_labels'])
    return(X, Y, train_idx, test_idx)
