clear all; %clc; %close all;
rng(1110,'twister');

% pre-process y
[x,y,xtest,ytest,y0] = read_fx();
[y,ymean,ystd] = standardize(y,[],[]);

%% ssvi
cf.covfunc_g  = 'covSEard';
cf.lrate      = 1e-2;
cf.momentum   = 0.9;
cf.lrate_hyp  = 1e-5;
cf.lrate_beta = 1e-4;
cf.momentum_w = 0.9;
cf.lrate_w    = 1e-5;
cf.learn_z    = false;
cf.momentum_z = 0.0;
cf.lrate_z    = 1e-4;
cf.init_kmeans = false;
cf.maxiter = 500;
cf.nbatch = 200;
cf.beta = 1/0.1;
cf.initz = 'random';

M = 100;
Q = 1;
par.g = cell(Q,1);
[elbo,par] = slfm2_learn(x,y,M,par,cf);
[mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,xtest,size(y,2));
mu =mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);

disp('elbo = ')
disp(elbo(end))
disp('learned w = ')
disp(par.w)

figure;
%plot(1:numel(elbo),elbo);
semilogy(1:numel(elbo),-elbo);
ylabel('elbo')
xlabel('iteration')
title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])

%%
outputs = [4,6,9];
titles = {'CAD','JPY','AUD'};
xtest = cell(3,1);
xtest{1} = (50:100)';
xtest{2} = (100:150)';
xtest{3} = (150:200)';
disp('smse of CAD, JPY, AUD')
theaxis{1} = [0 250 0.75 1.15];
theaxis{2} = [0 250 7.8*1e-3 9.4*1e-3];
theaxis{3} = [0 250 0.7 0.95];
smses = zeros(3,1);
for i=1:3
  t = outputs(i);
  plot_all(x,y0(:,t),x,mu(:,t),(ystd(t).^2)*var(:,t),[],[],titles{i});
  axis(theaxis{i});
  smses(i) = mysmse(ytest(xtest{i},t),mu(xtest{i},t),ymean(t));
end  
disp(smses)
disp('average = ')
disp(mean(smses))

