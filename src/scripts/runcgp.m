rand('twister', 1e6);
randn('state', 1e6);

%% select data set and convert to cgp format
dataset = 'juraCu';
log_transform = false;
switch dataset
  case 'juraCd'
  case 'juraCu'
    [x,y,xtest,ytest] = load_data('data/jura',dataset);
    features = 1:2;
  case 'concrete'
    [x,y,xtest,ytest] = load_data('data/concrete','concrete3');
    features = 3:5;
  case 'ms'
    [x,y,xtest,ytest] = load_data('data/ms','ms');
    [x,xmean,xstd] = standardize(x,[],[]);
    xtest = standardize(xtest,xmean,xstd);
    features = 1:size(x,2);
end

numOutputs = size(y,2);
XTemp = cell(1,numOutputs); yTemp = cell(1,numOutputs);
XTestTemp = cell(1,numOutputs); yTestTemp = cell(1,numOutputs);
for i=1:numOutputs
  XTemp{i} = x(:,features);
  yTemp{i} = y(:,i);
  if (strcmp(dataset, 'juraCd') || strcmp(dataset, 'juraCu')) && i > 1
    XTemp{i} = [XTemp{i}; xtest(:,features)];
    yTemp{i} = [yTemp{i}; ytest(:,i)];
  end
  if log_transform
    yTemp{i} = log(yTemp{i});
  end
  XTestTemp{i} = xtest(:,features);
  yTestTemp{i} = ytest(:,i);
end

scaleVal = zeros(1,size(yTemp, 2));
biasVal = zeros(1,size(yTemp, 2));
for k =1:size(yTemp, 2),
    biasVal(k) = mean(yTemp{k});
    scaleVal(k) = std(yTemp{k});
end

% cgp otpions
nlf = 2;
%ftc = exact; fitc,pitc,fic = approximation
options = multigpOptions('pitc'); % ftc = full training condition
options.initialInducingPositionMethod = 'espaced';
options.kernType = 'gg';
options.optimiser = 'scg';
options.nlf = nlf;
%options.beta = ones(1, size(yTemp, 2));
options.beta = 1;
options.bias =  [zeros(1, options.nlf) biasVal];
options.scale = [zeros(1, options.nlf) scaleVal];

q = size(XTemp{1}, 2); % input dimension
d = size(yTemp, 2) + options.nlf; % #number of outputs + #latent funcs

% containing data from both the latent functions and training/observed data
X = cell(d,1);
y = cell(d,1);

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
XTest = cell(d,1);

% all one for the latent functions
for j=1:options.nlf
    XTest{j} = ones(1, q);
end
for i = 1:size(yTemp, 2)
    XTest{i+options.nlf} = XTestTemp{i};
end

%% repeat
iters = 1;
display = 1;
max_iters = 50;
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
    mae_all(iter,i) = maerror;
    smse_all(iter,i) = smserror;
  end
end
disp('mae = ')
disp(mean(mae_all,1))
