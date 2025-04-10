import sklearn.linear_model as lm
from dtuimldmtools import bmplot, feature_selector_lr, rlr_validate
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from baseline import baseline_inner, baseline_outer
from ann import ann_inner, ann_outer

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

model = lm.LinearRegression()
model = model.fit(X, Y)

y_pred = model.predict(X)
residual = (y_pred-Y)*100/Y

plt.figure()
plt.subplot(2,1,1)
plt.plot(Y, y_pred, '.')
plt.xlabel('smoothness (true)')
plt.ylabel('smoothness (estimated)')
plt.subplot(2,1,2)
plt.hist(residual,30)

plt.show()


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
X_attributes = ["Offset"] + X_attributes
M = M + 1

Y=Y.squeeze()

## Crossvalidation

K = 10
CV = sklearn.model_selection.KFold(K,shuffle=True)

lambdas = np.power(10.0,range(-4,10))
#lambdas = np.array(range(-10000,20000,20))


# Initialize variables
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


k=0


for train_index, test_index in CV.split(X,Y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = Y[train_index]
    X_test = X[test_index]
    y_test = Y[test_index]
    internal_cross_validation = 10
    
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    lmbda = opt_lambda * np.eye(M)
    lmbda[0,0] = 0
    w_rlr[:,k] = np.linalg.solve(XtX+lmbda,Xty).squeeze()

    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))

    if k == K-1:
        plt.figure(k)
        plt.subplot(1,2,1)
        plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        plt.legend('Coefficients as function of regularization')
        plt.grid()

        plt.subplot(1,2,2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        plt.loglog(lambdas,test_err_vs_lambda.T,'r.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Generalization error')
        plt.legend('Generalization error as function of regularization')
        plt.grid()

    k+=1

plt.show()

# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print(m)
    print('{:>15} {:>15}'.format(X_attributes[m], np.round(w_rlr[m,-1],2)))

def linear_regression_inner(X_train_outer, y_train_outer, lambdas, K2):
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train_outer, y_train_outer, lambdas, K2)

    return opt_lambda

def linear_regression_outer(train_index_outer, test_index_outer, opt_lambda):

    X_train = X[train_index_outer]
    y_train = Y[train_index_outer]
    X_test = X[test_index_outer]
    y_test = Y[test_index_outer]

    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    lmbda = opt_lambda * np.eye(X.shape[1])
    lmbda[0,0] = 0

    w_rlr = np.linalg.solve(XtX + lmbda, Xty).squeeze()

    Error_train_rlr = (
        np.square(y_train - X_train @ w_rlr).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr = (
        np.square(y_test - X_test @ w_rlr).sum(axis=0) / y_test.shape[0]
    )

    return Error_test_rlr.mean()

