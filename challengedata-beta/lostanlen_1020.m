%% Load training features
X_train = load('X_train.mat');
X_train = sparse(X_train);
X_train = double(X_train);

%% Load training labels
Y_train = load('Y_train.mat');
Y_train = double(Y_train);

%% Load LIBLINEAR
addpath(genpath(('~/liblinear-2.1')));

%% Cross-validate regularization parameter C
solver_str = '-s 0'; % L1-regularized logistic regression
bias_str = '-B 1'; % a bias term is added
folds_str = '-v 5'; % 5-fold cross-validation is performed
reg_str = '-C';

str = [solver_str, bias_str, folds_str, reg_str];
validation_model = train(Y_train, X_train, str);
% Best is log2c=-6.00 with rate = 89.625%

be =

% (0) L2-regularized logistic regression (primal): 88.725%
% (6) L1-regularied logistic regression: 80,275%

%% Load testing features
X_test = load('X_test.mat');
X_test = sparse(X_test);
X_test = double(X_test);

%% Classify
