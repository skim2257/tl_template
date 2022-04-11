import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

from skimage.feature import hog
import time
import joblib
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

'''
parser = argparse.ArgumentParser()
parser.add_argument('--bin_num', nargs='?', const=5, type=int)
args = parser.parse_args()
print(args)

BIN_NUM = args.bin_num
print(BIN_NUM)
'''

BIN_NUM = 100
#if not severe then None == COVID
#TARGET = 'SEVERE'
TARGET = None


def get_accuracy(method, clf_list, clf_name, X_train, X_test, y_train, y_test):
    
    print(method + ' ANALYSIS')
    clf_list= []

    clf_list.append(method)
    clf_name.predict(X_train)
    train_acc = clf_name.score(X_train, y_train)
    clf_list.append(train_acc)
    print(method + ' Train Accuracy', train_acc)
    train_auroc =  roc_auc_score(y_train,clf_name.predict_proba(X_train)[:, 1])
    clf_list.append(train_auroc)
    print(method + ' Train AUROC',train_auroc)

    clf_name.predict(X_test)
    test_acc = clf_name.score(X_test, y_test)
    clf_list.append(test_acc)
    print(method + ' Test Accuracy', test_acc)
    test_auroc =  roc_auc_score(y_test,clf_name.predict_proba(X_test)[:, 1])
    clf_list.append(test_auroc)
    print(method + ' Test AUROC', test_auroc)
    return clf_list,clf_name.predict_proba(X_train),clf_name.predict_proba(X_test)

df_stoic = pd.read_csv(os.getenv('CLIN_PATH'), index_col=0)
df_train = df_stoic[df_stoic.split == "train"]
df_train['idx']=range(1600)

hist_pathname= os.getenv(F'HIST_{BIN_NUM}_PATH')

#joblib.load it back in
ptid,hist_vals, bin_vals,df_hist_train = joblib.load(hist_pathname)
hist_vals=np.array(hist_vals)

#X_train, X_test, y_train, y_test = train_test_split(hist_vals, labels, test_size=0.20, random_state=42)


#need to merge clinical variables with hist for top bin num
append_results = []
append_probabilities = []
for i in range(1,6):
    print(i)
    test_idx  = df_train[df_train['fold'] == np.float(i)].idx.to_numpy()
    train_idx = df_train[df_train['fold'] != np.float(i)].idx.to_numpy()


    X_train = hist_vals[train_idx]
    X_test  = hist_vals[test_idx]
    if TARGET =='SEVERE':
        y_train = df_train.probSevere.iloc[train_idx]
        y_test = df_train.probSevere.iloc[test_idx]
        print('Labels are probSEVERE')
    else:
        y_train = df_train.probCOVID.iloc[train_idx]
        y_test = df_train.probCOVID.iloc[test_idx]
        print('Labels are probCOVID')


    clf_LR = LogisticRegression(random_state=24, C=0.05, max_iter=1000).fit(X_train,y_train)
    lr_result, lr_probTrain, lr_probTest = get_accuracy('LOGISTIC REGRESSION', 'logreg', clf_LR,X_train, X_test, y_train, y_test)

    clf_SVM = SVC(kernel='rbf', random_state=24,C=0.05, probability=True).fit(X_train,y_train)
    svm_result,svm_probTrain, svm_probTest= get_accuracy('SVM', 'svm_list', clf_SVM,X_train, X_test, y_train, y_test)

    clf_KNN = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
    knn_result,knn_probTrain, knn_probTest= get_accuracy('KNN', 'knn_list', clf_KNN, X_train, X_test, y_train, y_test)

    clf_DT = DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
    DT_result,DT_probTrain, DT_probTest= get_accuracy('DT', 'DT_list', clf_DT, X_train, X_test, y_train, y_test)

    clf_RF= RandomForestClassifier(random_state=42, min_samples_split=8, max_depth=4).fit(X_train,y_train)
    RF_result,RF_probTrain, RF_probTest= get_accuracy('RF', 'RF_list', clf_RF, X_train, X_test, y_train, y_test)

    clf_MLP= MLPClassifier(random_state=42).fit(X_train,y_train)
    MLP_result,mlp_probTrain, mlp_probTest= get_accuracy('MLP', 'RF_list', clf_MLP, X_train, X_test, y_train, y_test)

    results_lists = [lr_result, svm_result,knn_result,DT_result,RF_result,MLP_result]
    columns= ['classifier', 'train_acc','train_auroc','test_accuracy','test_auroc']  
    df_results = pd.DataFrame(data=results_lists, columns=columns)
    append_results.append(df_results)
    
    clf_probabilities= [lr_probTest, svm_probTest, knn_probTest, DT_probTest, RF_probTest, mlp_probTest]
    append_probabilities.append(clf_probabilities)

df_probs = pd.DataFrame(append_probabilities)
df_final = pd.concat(append_results, ignore_index=True)

print('mean test auroc', np.mean(df_final.test_auroc))

import joblib

#hist_5_probabilities= [lr_result, lr_probTrain, lr_probTest,svm_result,svm_probTrain, svm_probTest,knn_result,knn_probTrain, knn_probTest,DT_result,DT_probTrain, DT_probTest,RF_result,RF_probTrain, RF_probTest,MLP_result,mlp_probTrain, mlp_probTest]



##oblib.dump([lr_result, lr_probTrain, lr_probTest,svm_result,svm_probTrain, svm_probTest,knn_result,knn_probTrain, knn_probTest,DT_result,DT_probTrain, DT_probTest,RF_result,RF_probTrain, RF_probTest,MLP_result,mlp_probTrain, mlp_probTest, y_train,y_test], eval_path, compress=1)

#joblib.load it back in
#lr_result, lr_probTrain, lr_probTest,svm_result,svm_probTrain, svm_probTest,knn_result,knn_probTrain, knn_probTest,DT_result,DT_probTrain, DT_probTest,RF_result,RF_probTrain, RF_probTest,MLP_result,mlp_probTrain, mlp_probTest,y_train,y_test = joblib.load(eval_path)


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_palette('pastel')
sns.set(rc={'figure.figsize':(10, 8)})


tst_aurocs = []
for i in range(6):
    z =np.concatenate(df_probs.iloc[:,i])
    tst_aurocs.append(z)

test_probs = np.array(tst_aurocs)
test_probs  = np.concatenate((test_probs, np.mean(test_probs, axis=0, keepdims=True)), axis=0)

if TARGET == 'SEVERE':
    y_test =np.array(df_train.probSevere)
else: 
    y_test =np.array(df_train.probCOVID)

## bootstrapping ##
n_bs = 250
p_bs = 1.

models   = ['LR', 'SVM', 'KNN', 'DT', 'RF', 'MLP', 'ensemble']
bs_test  = pd.DataFrame({'model': [], 'auroc': []})
for n, model in enumerate(models):
    y_hat = test_probs[n,:,0]
    bs_metrics = []
    for i in range(n_bs):
        idxs = np.random.choice(range(len(y_hat)), int(len(y_hat)*p_bs))
        bs_test = bs_test.append({'model': model, 'auroc': roc_auc_score(y_test[idxs], y_hat[idxs])}, ignore_index=True)
        
a = sns.violinplot(x='model', y='auroc', data=bs_test)
a.set_ylim(bottom=0.3, top=0.9)
if TARGET =='SEVERE':
    plt.title('Model Performance of Severity Prediction(Grey Level Histogram)')
else:
    plt.title('Model Performance of COVID Diagnosis (Grey Level Histogram)')
plt.show()       
fig = a.get_figure()
