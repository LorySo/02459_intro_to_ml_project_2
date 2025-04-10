import sklearn.linear_model as lm
from dtuimldmtools import bmplot, feature_selector_lr, rlr_validate, train_neural_net
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection
import torch
import scipy.stats
from baseline import baseline_inner, baseline_outer
from ann import ann_inner, ann_outer
from linear_regression import linear_regression_inner, linear_regression_outer


features = pd.read_csv("breast_cancer_wisconsin_features.csv")
targets = pd.read_csv("breast_cancer_wisconsin_targets.csv")

unique_attributes = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

for attr in unique_attributes:
    features = features.drop(columns=[attr+'2',attr+'3'], axis = 1)

# feature transformation

Y = features['smoothness1']

X = features.drop(columns=['smoothness1', 'perimeter1','area1','symmetry1'], axis = 1)
X_attributes = ['radius1', 'texture1', 'compactness1', 'concavity1', 'concave_points1','fractal_dimension']

X=X.to_numpy()
Y=Y.to_numpy()

N, M = X.shape

for i in range(M):
    mean = X[:, i].mean()
    std = X[:, i].std()

    X[:,i] = (X[:,i] - mean) / std


#X=X[targets['Diagnosis']=='M']
#Y=Y[targets['Diagnosis']=='M']

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
X_attributes = ["Offset"] + X_attributes
M = M + 1

Y=Y.squeeze()

lambdas = np.power(10.0,range(-4,10))
#lambdas = np.array(range(-10000,20000,20))


##1b

K1 = 10  # Outer CV
K2 = 10  # Inner CV

min_n_hidden_units = 1
max_n_hidden_units = 10
n_rep_ann = 3
max_iter = 1000

CV_1 = sklearn.model_selection.KFold(K1, shuffle=True)

threshold=0.05
fixed_lambda = 10
fixed_hidden_units = 1
n_rep_ann = 3
max_iter = 1000

baseline_preds = np.zeros(N)
rlr_preds = np.zeros(N)
ann_preds = np.zeros(N)

for (train_index_outer, test_index_outer) in CV_1.split(X):
    X_train_outer = X[train_index_outer]
    y_train_outer = Y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = Y[test_index_outer]
        
    # Baseline
    mean_y = np.mean(y_train_outer)
    y_est_test_outer_baseline = mean_y     
    test_error = np.sum((y_est_test_outer_baseline - y_test_outer)**2) / float(len(y_test_outer))
    
    baseline_preds[test_index_outer]=test_error

    # ANN
    X_train_outer_tensor = torch.tensor(X_train_outer, dtype=torch.float)
    y_train_outer_tensor = torch.tensor(y_train_outer, dtype=torch.float)
    X_test_outer_tensor = torch.tensor(X_test_outer, dtype=torch.float)
    y_test_outer_tensor = torch.tensor(y_test_outer, dtype=torch.float)
    
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, fixed_hidden_units),
        torch.nn.Tanh(),
        torch.nn.Linear(fixed_hidden_units, 1)
    )

    net, final_loss, learning_curve = train_neural_net(
        model, torch.nn.MSELoss(), X_train_outer_tensor, y_train_outer_tensor,
        n_replicates=n_rep_ann, max_iter=max_iter
    )

    y_est_outer = net(X_test_outer_tensor)
    e = (y_est_outer.float() - y_test_outer_tensor.float())**2
    error_rate = (sum(e).type(torch.float)/len(y_test_outer_tensor)).data.numpy()[0]
    
    ann_preds[test_index_outer] = error_rate
    
    # RLR
    rlr_preds[test_index_outer] = linear_regression_outer(train_index_outer, test_index_outer, fixed_lambda)


t_stat_blann, p_value_blann = scipy.stats.ttest_rel(baseline_preds, ann_preds)
print(t_stat_blann, p_value_blann)
t_stat_blrlr, p_value_blrlr = scipy.stats.ttest_rel(baseline_preds, rlr_preds)
print(t_stat_blrlr, p_value_blrlr)
t_stat_annrlr, p_value_annrlr = scipy.stats.ttest_rel(ann_preds, rlr_preds)
print(t_stat_annrlr, p_value_annrlr)

alpha = 0.05

bl_ann = baseline_preds-ann_preds
CI = scipy.stats.t.interval(1-alpha, len(bl_ann)-1, loc=np.mean(bl_ann), scale=scipy.stats.sem(bl_ann))
print("CI : ", np.round(CI, 15))

bl_rlr = baseline_preds-rlr_preds
CI = scipy.stats.t.interval(1-alpha, len(bl_rlr)-1, loc=np.mean(bl_rlr), scale=scipy.stats.sem(bl_rlr))
print("CI : ", np.round(CI, 15))

ann_rlr = ann_preds-rlr_preds
CI = scipy.stats.t.interval(1-alpha, len(ann_rlr)-1, loc=np.mean(ann_rlr), scale=scipy.stats.sem(ann_rlr))
print("CI : ", np.round(CI, 15))

