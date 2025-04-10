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
from dtuimldmtools import jeffrey_interval, mcnemar


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
hidden_units_ann = np.empty(K1)
gen_error_ann = np.empty(K1)
opt_lambdas_lr = np.empty(K1)
gen_error_lr = np.empty(K1)
gen_error_baseline = np.empty(K1)

test_error_outer_baseline = []
test_errors_outer_ANN = []
lambda_ANN = []


threshold=0.05
fixed_lambda = 10
fixed_hidden_units = 1
n_rep_ann = 3
max_iter = 1000

K = 10 

baseline_preds = np.zeros(N)
rlr_preds = np.zeros(N)
ann_preds = np.zeros(N)

for k_outer, (train_index_outer, test_index_outer) in enumerate(CV_1.split(X)):
    X_train_outer = X[train_index_outer]
    y_train_outer = Y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = Y[test_index_outer]

    CV_2 = sklearn.model_selection.KFold(K2, shuffle=True)
    
    error_inner_baseline = []
    error_inner_ann_matrix = []
    hidden_units_matrix = []
        
    # Baseline predictions
    mean_y = np.mean(y_train_outer)
    baseline_preds[test_index_outer] = mean_y

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

    y_ann = net(X_test_outer_tensor)
    e = (y_ann.float() - y_test_outer_tensor.float())**2
    error_rate = (sum(e).type(torch.float)/len(y_test_outer_tensor)).data.numpy()[0]
    
    ann_preds[test_index_outer]=error_rate

    # RLR
    model = lm.LinearRegression()
    model = model.fit(X_train_outer, y_train_outer)

    rlr_preds[test_index_outer]  = model.predict(X_train_outer)

# Jeffrey Interval and McNemar's Test
[thetahat_baseline, CIA_baseline] = jeffrey_interval(Y, baseline_preds, alpha=threshold)
[thetahat_ann, CIA_ann] = jeffrey_interval(Y, ann_preds, alpha=threshold)
[thetahat_rlr, CIA_rlr] = jeffrey_interval(Y, rlr_preds, alpha=threshold)

print(f"Baseline, Thetahat: {thetahat_baseline}, CI: {CIA_baseline}")
print(f"ANN, Thetahat: {thetahat_ann}, CI: {CIA_ann}")
print(f"RLR, Thetahat: {thetahat_rlr}, CI: {CIA_rlr}")

# Pair evaluations for McNemar's test
observation_pairs = [
    (baseline_preds, ann_preds),
    (baseline_preds, rlr_preds),
    (ann_preds, rlr_preds)
]
pairlist = ["baseline-ann", "baseline-rlr", "ann-rlr"]

for index, pair in enumerate(observation_pairs):
    obs1, obs2 = pair
    print(f"\nEvaluating pair: {pairlist[index]}")
    thetahat, CI, p_value = mcnemar(Y, obs1, obs2, alpha=threshold)
    print(f"McNemar result for {pairlist[index]}: Thetahat = {thetahat}, CI = {CI}, p-value = {p_value}")
