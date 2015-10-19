%% Load training features
X_train = csvread('X_train.csv');
X_train = single(X_train);

%% Load training labels
uiopen('/Users/vlostan/enschallengedata/Y_train.csv',1);
Y_train = Target;

%% Extract samples that correspond to classes 0 and 1
X0_train = X_train(Y_train==0, :);
X1_train = X_train(Y_train==1, :);

%% Sort feature vectors according to class
X01_train = cat(1, X0_train, X1_train);

%% Compute pairwise distances
pdists_vec = pdist(X01_train.');

%% Put pairwise distances in square form
pdists_square = squareform(pdists_vec);

%%
