%% Load training features
X_train = load('X_train.mat');
X_train = double(X_train);

%% Load training labels
Y_train = load('Y_train.mat');
Y_train = double(Y_train);

%% Load testing features
X_test = load('X_test.mat');
X_test = double(X_test);

%% Scale features
X = cat(1, X_train, X_test);
mu = mean(X);
sigma = std(X);

centeredX_train = bsxfun(@minus, X_train, mu);
stdX_train = bsxfun(@rdivide, centeredX_train, sigma);

centeredX_test = bsxfun(@minus, X_test, mu);
stdX_test = bsxfun(@rdivide, centeredX_test, sigma);

spX_train = sparse(X_train);
spX_test = sparse(X_test);
spY_train = double(Y_train);

%% Load LIBLINEAR
addpath(genpath(('~/liblinear-2.1')));

%% Cross-validate regularization parameter C
solver_str = '-s 0'; % L1-regularized logistic regression
bias_str = '-B 1'; % a bias term is added
folds_str = '-v 5'; % 5-fold cross-validation is performed
reg_str = '-C';

validation_str = [solver_str, bias_str, folds_str, reg_str];
validation_model = train(Y_train, X_train, validation_str);
% Best is log2c=-6.00 with rate = 89.625%

%% Train on best model
cost_str = '-c 0.015625'
train_str = [solver_str, bias_str, folds_str, cost_str];
train_model = train(Y_train, X_train, train_str);

%% Load testing features
X_test = load('X_test.mat');
X_test = sparse(X_test);
X_test = double(X_test);

%% Classify
