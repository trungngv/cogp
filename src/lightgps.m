% Comparison of different light gp models. Note though that it is hard to
% compare them in an absolutely fair way. The reason being the covariance
% functions (hence effectively the GPs) are of different forms. Perhaps the
% best comparison is to evaluate their average performance over multpile
% runs.
%
% The spare models include:
% - gp with fitc approximation (snelson)
% - convolved gp with discrete approximation (higdon & myself)
% - convolved gp with latent independence (alvarez)
%
% Performance depends on:
%   - covariance function (different covariance functions are created)
%   - transformation of outputs (e.g. zero mean) or inputs (scaling etc)
%   - initial inducing inputs (Xu) and whether they are learned (optimized)
%   - initial value of hypers (theta) and whether they are learned
%   (optimized)
% 
% Trung V. Nguyen
% 18/07/2013

clear; close all;
% test data
x = load('train_inputs');
y = load('train_outputs');
ymean = mean(y);
y0 = y - ymean; % zero-mean for all methods
xt = load('test_inputs');

% inducing inputs
num_induc = 10;
[induc_indices, xu] = select_inducing('r', x, num_induc);

%% spgp
[N,dim] = size(x);
hyp_init(1:dim,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
hyp_init(dim+1,1) = log(var(y0,1)); % log size 
hyp_init(dim+2,1) = log(var(y0,1)/4); % log noise

%% cgp
nTasks = size(y, 2);
Xtrain = cell(1,nTasks); Ytrain = cell(1,nTasks);
Xtest = cell(1,nTasks); Ytest = cell(1,nTasks);
for task = 1:nTasks
  Xtrain{task} = x;  Ytrain{task} = y(:,task);
  Xtest{task} = xt;
  %Ytest{task} = yt(task,:)';
end

options = multigpOptions('fitc'); % fully independent training conditional
options.kernType = 'gg'; % also the default
options.optimiser = 'scg'; % also the default
options.nlf = 1; % also the default
%options.learnScales = false;
options.numActive = num_induc;
options.fixInducing = false;
options.fixIndices = induc_indices;
options.initialInducingPositionMethod = 'fixIndices';
%options.beta = ones(1, size(Ytrain, 2));

q = size(Xtrain{1}, 2); % input dimension (spatical coordinates)
d = nTasks; % d = #number of outputs

% see demSpmgpGgToy1 for example init
Xcell = cell(nTasks, 1); ycell = cell(nTasks, 1);

for i = 1:nTasks
    ycell{i} = Ytrain{i};    Xcell{i} = Xtrain{i};
end

% Creates the model
model = multigpCreate(q, d, Xcell, ycell, options);
disp('initial model')
multigpDisplay(model);

display = 0; iters = 100;
% learn model parameters from the training data (X, y)
model = multigpOptimise(model, display, iters);
disp('learned model')
multigpDisplay(model);

% Prediction for all
multigpXtest = cell(nTasks, 1);
for i = 1:nTasks
    multigpXtest{i} = Xtest{i};
end

[mu vars]= multigpPosteriorMeanVar(model, multigpXtest);

figure; hold on;
% mu{1} and vars{1} is for latent function u(x)
plotMeanAndStd(multigpXtest{i}, mu{2}, sqrt(vars{2}), [7 7 7] / 8);
xlim([min([x; xt]), max([x; xt])]);
%plot(multigpXtest{i}, mu{i}, ['' color(i)], 'MarkerSize', 1.6);
title('Prediction by convolved GP');

plot(x,y,'.');
xlabel('x');
ylabel('y');
