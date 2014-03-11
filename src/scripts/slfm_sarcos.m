clear all; %clc; %close all;

% pre-process y
load sarcos_inv
load sarcos_inv_test
fmae = @(ytrue,ypred) mean(abs(ypred-ytrue));
frmse = @(ytrue,ypred) sqrt(mean((ypred-ytrue).^2));
x = sarcos_inv(:,1:21);
y = sarcos_inv(:,22:end);

xtest = sarcos_inv_test(:,1:21);
ytest = sarcos_inv_test(:,22:end);
outputs = [4,7];
y = y(:,outputs);
ytest = ytest(:,outputs);
[x,xmean,xstd] = standardize(x,[],[]);
xtest = standardize(xtest,xmean,xstd);
[y,ymean,ystd] = standardize(y,[],[]);
clear sarcos_inv
clear sarcos_inv_test

%% ssvi
cf.covfunc_g  = 'covSEard';
cf.covfunc_h = 'covSEard';
cf.lrate      = 1e-2;
cf.momentum   = 0.9;
cf.lrate_hyp  = 1e-5;
cf.lrate_beta = 1e-4;
cf.momentum_w = 0.9;
cf.lrate_w    = 1e-6;
cf.learn_z    = true;
cf.momentum_z = 0.0;
cf.lrate_z    = 1e-4;
cf.maxiter = 1;
cf.nbatch = 2000;
cf.beta = 1/0.01;
cf.initz = 'random';
cf.monitor_elbo = 50;
cf.w = ones(size(y,2),1);
%cf.w = [1 0.01; 0.01 1];
cf.fix_first = true;

Q = 1;
M.g = 500; M.h = 500;
runs = 5;
smses = zeros(2,1);
maes = smses;
nlpds = smses;
elbos = zeros(runs,1);
runningTime = elbos;
for r=1:runs
  rng(1110+r,'twister');
  par.g = cell(Q,1); par.task = cell(size(y,2),1);
  % par.task{1}.loghyp = log(0.1);
  % par.task{2}.loghyp = log(0.1);
  % M = 500; [elbo,par] = slfm2_learn(x,y,M,par,cf);
  % [mu,vaar,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,xtest,size(y,2));
  tstart = tic;
  [elbo,par] = slfm_learn(x,y,M,par,cf);
  runningTime(r) = toc(tstart);
  elbos(r) = elbo(end);
  
  [mu,vaar,mu_g,var_g] = slfm_predict(cf.covfunc_g,cf.covfunc_h,par,xtest);
  mu = mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
  fvar = vaar.*repmat(ystd.^2,size(mu,1),1);
  for i=1:2
    maes(r,i) = mean(abs(ytest(:,i) - mu(:,i)));
    smses(r,i) = mysmse(ytest(:,i),mu(:,i),ymean(i));
    nlpds(r,i) = mynlpd(ytest(:,i),mu(:,i),fvar(:,i));
  end
  save('sarcos-cogp');
  disp('test mae, smse, nlpd = ')
  disp(maes(r,:))
  disp(smses(r,:))
  disp(nlpds(r,:))
end

% figure;
% %plot(1:numel(elbo),elbo);
% elbo = elbo(elbo ~= 0);
% semilogy(1:numel(elbo),-elbo);
% ylabel('elbo')
% xlabel('iteration')
% title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])

% disp('learned w = ')
% disp(par.w)
