clear all; clc; %close all;
rng(1110,'twister');

[x,y,xtest,ytest] = read_juraCd();
% pre-process y
y0 = y;
y = log(y);
[y,ymean,ystd] = standardize(y,[],[]);

% ssvi
cf.covfunc_g  = 'covSEard';
cf.lrate      = 1e-2;
cf.momentum   = 0.9;
cf.lrate_hyp  = 1e-5;
cf.lrate_beta = 1e-4;
cf.momentum_w = 0.9;
cf.lrate_w    = 1e-4;
cf.learn_z    = false;
cf.momentum_z = 0.0;
cf.lrate_z    = 1e-4;
cf.init_kmeans = false;
cf.maxiter = 2000;
cf.nbatch = 100;

Q = 2;
runs = 10;
Ms = [5,10,20,50,100,150,200,259];
maes = zeros(numel(Ms),runs);
for i=1:numel(Ms)
  M = Ms(i);
  for j=1:runs
    par.g = cell(Q,1);
    [elbo,par] = slfm2_learn(x,y,M,par,cf);
    [mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,xtest,size(y,2));
    mu = exp(mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1));
    maes(i,j) = mean(abs(ytest(:,1) - mu(:,1)));
    clear par;
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

% par.g = cell(Q,1);
% [elbo,par] = slfm2_learn(x,y,M,par,cf);
% [mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,xtest,size(y,2));
% mu = exp(mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1));
% disp('mae test = ')
% disp(mean(abs(mu-ytest)))

% disp('mae train = ')
% [mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,x,size(y,2));
% mu = exp(mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1));
% y0(260:end,1) = ytest(:,1);
% disp(mean(abs(mu-y0)));

% figure;
% %plot(1:numel(elbo),elbo);
% semilogy(1:numel(elbo),-elbo);
% ylabel('elbo')
% xlabel('iteration')
% title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])

% disp('elbo = ')
% disp(elbo(end))
% disp('learned w = ')
% disp(par.w)

% figure; hold on;
% scatter(x(:,1),x(:,2))
% scatter(par.g{1}.z(:,1),par.g{1}.z(:,2),'xr')
% scatter(par.g{1}.z0(:,1),par.g{1}.z0(:,2),'sr')
% scatter(par.g{2}.z(:,1),par.g{2}.z(:,2),'xm')
% scatter(par.g{2}.z0(:,1),par.g{2}.z0(:,2),'sm')
% title('inducing of g')

% standard full gp or standard gpsvi?

