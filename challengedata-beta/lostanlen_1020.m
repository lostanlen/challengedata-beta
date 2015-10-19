%% Load training features
X_train = load('X_train.mat');
X_train = sparse(X_train);
X_train = double(X_train);

%% Load training labels
Y_train = load('Y_train.mat');
Y_train = double(Y_train);

%% Load LIBLINEAR
addpath(genpath(('~/liblinear-2.1'));

%% Train linear classifier
model = train(Y_train, X_train);

%% Load testing features
X_test = load('X_test.mat');
X_test = sparse(X_test);
X_test = double(X_test);
