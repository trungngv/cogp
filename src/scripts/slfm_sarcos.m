clear all; clc; %close all;
rng(1110,'twister');

% pre-process y
load sarcos_inv
load sarcos_inv_test
fmae = @(ytrue,ypred) mean(abs(ypred-ytrue));
frmse = @(ytrue,ypred) sqrt(mean((ypred-ytrue).^2));

x = sarcos_inv(:,1:21);
y = sarcos_inv(:,22:end);
xtest = sarcos_inv_test(:,1:21);
ytest = sarcos_inv_test(:,22:end);
outputs = [2,3,4,7];
y = y(:,outputs);
ytest = ytest(:,outputs);
[x,xmean,xstd] = standardize(x,[],[]);
xtest = standardize(xtest,xmean,xstd);
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
cf.maxiter = 100;
cf.nbatch = 500;
cf.beta = 1/1;
cf.initz = 'kmeans';

Q = 1;
runs = 1;
Ms = [100,200,500,700,1000];
Ms = 500;
for i=1:numel(Ms)
  M = Ms(i);
  disp(['M = ' num2str(M)])
  for j=1:runs
    par.g = cell(Q,1);
    [elbo,par] = slfm2_learn(x,y,M,par,cf);
    [mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,xtest,size(y,2));
    mu = mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
    clear par;
    disp('mae test = ')
    disp(fmae(ytest,mu))
    disp('rmse test = ')
    disp(frmse(ytest,mu))
  end
end

mae_mean = mean(maes(1:4,:),2);
mae_std = 2*std(maes(1:4,:),0,2);
plot(Ms(1:4),mae_mean,'x-','MarkerSize',15);
errorbar(Ms(1:4),mae_mean,mae_std);
xlabel('number of inducing points');
ylabel('mae');
title('slfm fixed z, mae vs. inducing inputs');
saveas(gcf, 'juraCd-slfm.eps','epsc');
save('juraCd-slfm.mat','maes');

% M = 200;
% Q = 2;
% par.g = cell(Q,1);
% [elbo,par] = slfm2_learn(x,y,M,par,cf);
% [mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,xtest,size(y,2));
% mu =mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
% disp('mae test = ')
% disp(fmae(ytest,mu))
% disp('rmse test = ')
% disp(frmse(ytest,mu))

figure;
%plot(1:numel(elbo),elbo);
semilogy(1:numel(elbo),-elbo);
ylabel('elbo')
xlabel('iteration')
title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])

disp('elbo = ')
disp(elbo(end))
disp('learned w = ')
disp(par.w)

