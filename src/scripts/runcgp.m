rand('twister', 1e6);
randn('state', 1e6);

%% select data set and convert to cgp format
[x,y,xtest,ytest,y0] = read_weather();
[~,ymean,~] = standardize(y,[],[]);
test_ind{1} = x >= 10.2 & x <= 10.8;
test_ind{2} = x >= 13.5 & x <= 14.2;

numOutputs = size(y,2);
XTemp = cell(1,numOutputs); yTemp = cell(1,numOutputs);
XTestTemp = cell(1,numOutputs); yTestTemp = cell(1,numOutputs);
for i=1:numOutputs
  % missing data here
  indi = ~isnan(y(:,i));
  XTemp{i} = x(indi,:);
  yTemp{i} = y(indi,i);
  XTestTemp{i} = xtest;
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
options = multigpOptions('ftc'); % ftc = full training condition
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
runs = 5;
display = 1;
max_iters = 100;
outputs = [2,3];
mae_all = zeros(runs,2);
smse_all = mae_all;
nlpd_all = mae_all;
for iter=1:runs
  model = multigpCreate(q, d, X, y, options);
  %multigpDisplay(model);
  %pause
  tstart = tic;
  [model, ~, funcval] = multigpOptimise(model, display, max_iters);
  toc(tstart)
  
  % Prediction
  [mu vaar] = multigpPosteriorMeanVar(model, XTest);
  for i = 1:2
    t = outputs(i);
    tmp = ~isnan(ytest(:,t)) & test_ind{i};
    yytest = ytest(tmp,t);
    yypred = mu{model.nlf + t}(tmp);
    yvar = vaar{model.nlf + t}(tmp);
    mae_all(iter,i) = mean(abs((yytest - yypred)));
    smse_all(iter,i) = mysmse(yytest,yypred,ymean(t));
    nlpd_all(iter,i) = mynlpd(yytest,yypred,yvar);
    save('weather-cgp');
  end
  disp('smse = ')
  disp(smse_all(iter,:))
end
disp('avg smse = ')
disp(mean(smse_all,1))

% use this to make the plots
% plot_all(x,y0(:,2),x,mu{model.nlf+2},2*sqrt(var{model.nlf+2}),[],[],'cgp')
% axis([10 15 10 28])
% plot_all(x,y0(:,3),x,mu{model.nlf+3},2*sqrt(var{model.nlf+2}),[],[],'cgp')
% axis([10 15 10 28])

