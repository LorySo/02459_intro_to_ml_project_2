import sklearn.linear_model as lm
from dtuimldmtools import bmplot, feature_selector_lr, rlr_validate, train_neural_net
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection
import torch
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

K1 = 10  # Outer
K2 = 10  # Inner

min_n_hidden_units = 1
max_n_hidden_units = 10
n_rep_ann = 3
max_iter = 1000

CV_1 = sklearn.model_selection.KFold(K1, shuffle=True)
hidden_units_ann = np.empty(K1)
gen_error_ann = np.empty(K1)
opt_lambdas_lr = np.empty(K1)
gen_error_lr = np.empty(K1)
gen_error_baseline = np.empty(K1)

test_error_outer_baseline = []
test_errors_outer_ANN = []
lambda_ANN = []

for k_outer, (train_index_outer, test_index_outer) in enumerate(CV_1.split(X)):
    X_train_outer = X[train_index_outer]
    y_train_outer = Y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = Y[test_index_outer]

    CV_2 = sklearn.model_selection.KFold(K2, shuffle=True)
    error_inner_baseline = []
    data_validation_length = []
    error_inner_ann_matrix = []
    hidden_units_matrix = []

    for k_inner, (train_index_inner, test_index_inner) in enumerate(CV_2.split(X_train_outer)):
        X_train_inner = X_train_outer[train_index_inner]
        y_train_inner = y_train_outer[train_index_inner]
        X_test_inner = X_train_outer[test_index_inner]
        y_test_inner = y_train_outer[test_index_inner]

        # Baseline
        error_inner_baseline, data_validation_length = baseline_inner(y_train_inner, y_test_inner, error_inner_baseline, data_validation_length)

        # ANN
        X_train_inner_tensor = torch.tensor(X_train_inner, dtype=torch.float)
        y_train_inner_tensor = torch.tensor(y_train_inner, dtype=torch.float)
        X_test_inner_tensor = torch.tensor(X_test_inner, dtype=torch.float)
        y_test_inner_tensor = torch.tensor(y_test_inner, dtype=torch.float)

        errors, hidden_units = ann_inner(
            X_train_inner_tensor, y_train_inner_tensor,
            X_test_inner_tensor, y_test_inner_tensor,
            min_n_hidden_units, max_n_hidden_units,
            M, n_rep_ann, max_iter
        )

        error_inner_ann_matrix.append(errors)
        hidden_units_matrix.append(hidden_units)


    # Baseline outer
    test_error_outer_baseline = baseline_outer(y_train_outer, y_test_outer, test_error_outer_baseline)

    # ANN outer
    X_train_outer_tensor = torch.tensor(X_train_outer, dtype=torch.float)
    y_train_outer_tensor = torch.tensor(y_train_outer, dtype=torch.float)
    X_test_outer_tensor = torch.tensor(X_test_outer, dtype=torch.float)
    y_test_outer_tensor = torch.tensor(y_test_outer, dtype=torch.float)

    optimal_hidden_units_ANN, test_errors_outer_ANN = ann_outer(
        np.array(hidden_units_matrix),
        np.array(error_inner_ann_matrix),
        data_validation_length,
        len(y_train_outer),
        X_train_outer_tensor,
        y_train_outer_tensor,
        X_test_outer_tensor,
        y_test_outer_tensor,
        k_outer,
        test_errors_outer_ANN,
        M,
        n_rep_ann,
        max_iter
    )

    # RLR
    lmbda= linear_regression_inner(X_train_outer, y_train_outer, lambdas, K2)
    error_rlr = linear_regression_outer(train_index_outer, test_index_outer, lmbda)
    
    print(k_outer)
    print(f"test error baseline : {test_error_outer_baseline}")
    print(f"test error ann : {test_errors_outer_ANN}")
    print(f"test error RLR : {error_rlr}")
    print(f"lambda RLR : {lmbda}")
    print(f"Unités cachées optimales : {optimal_hidden_units_ANN}")
