% CGP_PREDICTION Demonstrate multigp convolution model.
% 
% Requires library in libs/cgp.
%
%	Description:
%	the FULL covariance matrix.
% 	demGgJura.m SVN version 312
% 	last update 2009-04-08T10:39:37.000000Z

rng(100, 'twister');

% XTemp, yTemp = training
% XTestTemp, yTestTemp = validation
% think of them as X, y, XTest, yTest
basedir = 'data';
basefile = 'gprn31';
[x, y, xt, yt] = load_data(basedir, basefile);

% Convert the data format to that used by multigp implementation
% from current data set: XTemp: Ndata x D, yTemp: Q x Ndata, similarly for
% XTestTemp, yTestTemp
nTasks = size(y, 1);
Xtrain = cell(1,nTasks);
Ytrain = cell(1,nTasks);
Xtest = cell(1,nTasks);
Ytest = cell(1,nTasks);
for task = 1:nTasks
  Xtrain{task} = x;
  Ytrain{task} = y(task,:)';
  Xtest{task} = xt;
  Ytest{task} = yt(task,:)';
end

scaleVal = zeros(1,nTasks);
biasVal = zeros(1,nTasks);
for k =1:nTasks,
    biasVal(k) = mean(Ytrain{k});
    scaleVal(k) = sqrt(var(Ytrain{k}));
end

options = multigpOptions('ftc'); % full training condition
options = multigpOptions('fitc'); % fully independent training conditional
options.kernType = 'gg';
options.optimiser = 'scg';
options.nlf = 1;
options.beta = ones(1, size(Ytrain, 2));
options.bias =  [zeros(1, options.nlf) biasVal];
options.scale = [zeros(1, options.nlf) scaleVal];

q = size(Xtrain{1}, 2); % input dimension (spatical coordinates)
d = nTasks + options.nlf; % 4 = #number of outputs + #latent funcs

% containing data from both the latent functions and training/observed data
X = cell(nTasks+options.nlf, 1);
y = cell(nTasks+options.nlf, 1);

% from 1 to options.nlf are indice of latent functions data
for j=1:options.nlf
    y{j} = [];
    X{j} = zeros(1, q); % no inducing value (for ftc)
end
% from nlf to last are indice of observed data
for i = 1:nTasks
    y{i+options.nlf} = Ytrain{i};
    X{i+options.nlf} = Xtrain{i};
end

% Creates the model
model = multigpCreate(q, d, X, y, options);
%multigpDisplay(model);

display = 1;
iters = 100;
% learn model parameters from the training data (X, y)
model = multigpOptimise(model, display, iters);
%multigpDisplay(model);

% Prediction for all
% Similarly for testing data
multigpXtest = cell(nTasks + options.nlf, 1);
% all one for the latent functions (does not really matter)
for j=1:options.nlf
    multigpXtest{j} = ones(1, q); % should be empty!
end
for i = 1:nTasks
    multigpXtest{i + options.nlf} = [Xtrain{i}; Xtest{i}];
end

mu = multigpPosteriorMeanVar(model, multigpXtest);
figure(31); hold on;
color = 'bgmr';
for i = 1:nTasks
  %maerror = mean(abs((Ytest{i} - mu{model.nlf + i})))
  plot(multigpXtest{model.nlf + i}, mu{model.nlf + i}, ['o' color(i)], 'MarkerSize', 1.6);
end
title('Prediction by convolved GP');
