rand('twister', 1e6);
randn('state', 1e6);

%% select data set and convert to cgp format
dataset = 'jura';
log_transform = true;
[x,y,xtest,ytest] = load_data('data/jura', [dataset 'Cd']);
numOutputs = 3;
XTemp = cell(1,numOutputs); yTemp = cell(1,numOutputs);
XTestTemp = cell(1,numOutputs); yTestTemp = cell(1,numOutputs);
if strcmp(dataset, 'jura')
  features = 1:2;
elseif strcmp(dataset, 'concrete')
  features = 3:5;
end
for i=1:numOutputs
  XTemp{i} = x(:,features);
  yTemp{i} = y(:,i);
  if strcmp(dataset, 'jura') && i > 1
    XTemp{i} = [XTemp{i}; xtest(:,features)];
    yTemp{i} = [yTemp{i}; ytest(:,i)];
  end
  if log_transform
    yTemp{i} = log(yTemp{i});
  end
  XTestTemp{i} = xtest(:,features);
  yTestTemp{i} = ytest(:,i);
end

%%
scaleVal = zeros(1,size(yTemp, 2));
biasVal = zeros(1,size(yTemp, 2));
for k =1:size(yTemp, 2),
    biasVal(k) = mean(yTemp{k});
    scaleVal(k) = std(yTemp{k});
end

nlfs = [2];
for nlf=nlfs
% ftc = full training condition
options = multigpOptions('ftc');
options.kernType = 'gg';
options.optimiser = 'scg';
options.nlf = nlf;
options.beta = ones(1, size(yTemp, 2));
options.bias =  [zeros(1, options.nlf) biasVal];
options.scale = [zeros(1, options.nlf) scaleVal];

q = size(XTemp{1}, 2); % input dimension (spatical coordinates)
d = size(yTemp, 2) + options.nlf; % 4 = #number of outputs + #latent funcs

% containing data from both the latent functions and training/observed data
X = cell(size(yTemp, 2)+options.nlf,1);
y = cell(size(yTemp, 2)+options.nlf,1);

% from 1 to options.nlf are indice of latent functions data
for j=1:options.nlf
    y{j} = [];
    X{j} = zeros(1, q); % no inducing value (for ftc)
end
% from nlf to last are indice of observed data
for i = 1:size(yTemp, 2)
    y{i+options.nlf} = yTemp{i};
    X{i+options.nlf} = XTemp{i};
end

% Similarly for testing data
XTest = cell(size(yTemp, 2)+options.nlf,1);

% all one for the latent functions
for j=1:options.nlf
    XTest{j} = ones(1, q);
end
for i = 1:size(yTemp, 2)
    XTest{i+options.nlf} = XTestTemp{i};
end

%% model selection
iters = 10;
display = 1;
max_iters = 1000;
mae_all = zeros(iters,numOutputs);
smse_all = mae_all;
funcval_all = zeros(iters,1);
model_all = cell(iters,1);
for iter=1:iters
model = multigpCreate(q, d, X, y, options);
model_all{iter} = model;
%multigpDisplay(model);
%pause

[model, ~, funcval] = multigpOptimise(model, display, max_iters);
funcval_all(iter) = funcval;

% Prediction
mu = multigpPosteriorMeanVar(model, XTest);
for i = 1:length(yTestTemp)
  if log_transform
    maerror = mean(abs((yTestTemp{i} - exp(mu{model.nlf + i}))));
    smserror = mean((yTestTemp{i} - exp(mu{model.nlf + i})).^2)/(var(yTestTemp{i}) + mean(yTestTemp{i})^2);
  else
    maerror = mean(abs((yTestTemp{i} - mu{model.nlf + i})));
    smserror = mean((yTestTemp{i} - mu{model.nlf + i}).^2)/(var(yTestTemp{i}) + mean(yTestTemp{i})^2);
  end  
  mae_all(iter,i) = maerror
  smse_all(iter,i) = smserror
end
end
if log_transform
  save(['output/npvgprn/' dataset '/logcgp_nlf' num2str(options.nlf) '_P' num2str(numOutputs) '.mat'], 'mae_all', 'smse_all');
else
  save(['output/npvgprn/' dataset '/cgp_nlf' num2str(options.nlf) '_P' num2str(numOutputs) '.mat'], 'mae_all', 'smse_all');
end  

%% run after model selection
iters = 10;
display = 1;
max_iters = 1000;
mae_all = zeros(iters,numOutputs);
smse_all = mae_all;
[~, best_idx] = min(funcval_all);
for iter=1:iters
model = model_all{best_idx};
model = multigpOptimise(model, display, max_iters);
mu = multigpPosteriorMeanVar(model, XTest);
for i = 1:length(yTestTemp)
  if log_transform
    maerror = mean(abs((yTestTemp{i} - exp(mu{model.nlf + i}))));
    smserror = mean((yTestTemp{i} - exp(mu{model.nlf + i})).^2)/(var(yTestTemp{i}) + mean(yTestTemp{i})^2);
  else
    maerror = mean(abs((yTestTemp{i} - mu{model.nlf + i})));
    smserror = mean((yTestTemp{i} - mu{model.nlf + i}).^2)/(var(yTestTemp{i}) + mean(yTestTemp{i})^2);
  end  
  mae_all(iter,i) = maerror
  smse_all(iter,i) = smserror
end
end
if log_transform
  save(['output/npvgprn/' dataset '/logcgp_nlf' num2str(options.nlf) '_P' num2str(numOutputs) '_selection.mat'], 'mae_all', 'smse_all');
else
  save(['output/npvgprn/' dataset '/cgp_nlf' num2str(options.nlf) '_P' num2str(numOutputs) '_selection.mat'], 'mae_all', 'smse_all');
end
end
