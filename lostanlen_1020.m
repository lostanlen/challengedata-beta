%% Load training features
X_train = load('X_train.mat');
X_train = X_train.X_train;
X_train = double(X_train);

%% Load training labels
Y_train = load('Y_train.mat');
Y_train = Y_train.Y_train;
Y_train = double(Y_train);

%% Load testing features
X_test = load('X_test.mat');
X_test = X_test.X_test;
X_test = double(X_test);

%% Scale features
X = cat(1, X_train, X_test);
mu = mean(X);
sigma = std(X);

centeredX_train = bsxfun(@minus, X_train, mu);
stdX_train = bsxfun(@rdivide, centeredX_train, sigma);

centeredX_test = bsxfun(@minus, X_test, mu);
stdX_test = bsxfun(@rdivide, centeredX_test, sigma);

%% Convert to sparse matrices
spX_train = sparse(stdX_train);
spX_test = sparse(stdX_test);

%% Load LIBLINEAR
addpath(genpath(('~/liblinear-2.1')));

%% Cross-validate regularization parameter C
solver_str = '-s 0 '; % L2-regularized logistic regression
bias_str = '-B 1 '; % a bias term is added
folds_str = '-v 5 '; % 5-fold cross-validation is performed
reg_str = '-C '; % cost is optimized by cross-validation

validation_str = [solver_str, bias_str, folds_str, reg_str];
validation_model = train(Y_train, spX_train, validation_str);
% Best is log2c = -6.00 with rate = 91.625%

%% Train on best model
cost_str = '-c 0.015625'
train_str = [solver_str, bias_str, cost_str];
model = train(Y_train, spX_train, train_str);

%% Classify
surrogate_Y_test = zeros(4000, 1);
predicted_Y_test = predict(surrogate_Y_test, spX_test, model);

%% Export to CSV
firstcol_ids = arrayfun(@num2str, (4001:8000).', 'UniformOutput', false);
firstcol = cellfun(@(x) ['Id', x], firstcol_ids, 'UniformOutput', false);
secondcol = ...
    arrayfun(@num2str, summary.predicted_Y_test, 'UniformOutput', false);
csv_matrix = [firstcol, secondcol];
csvwrite('lostanlen_predicted.csv', csv_matrix);

%% Export summary for reproducibility
summary = struct( ...
    'model', model, ...
    'predicted_Y_test', predicted_Y_test, ...
    'train_str', train_str, ...
    'validation_str', validation_str);

save('summary', 'summary');
