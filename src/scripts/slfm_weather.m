%% reading data and model setting
clear all; %clc; %close all;
rng(1110,'twister');

%explorary analysis
[x,y,xtest,ytest,y0] = read_weather();

% pre-process y
[y,ymean,ystd] = standardize(y,[],[]);

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
cf.maxiter = 1500;
cf.nbatch = 1000;
cf.beta = 1/0.1;
cf.initz = 'random';
cf.w = ones(size(y,2),2);
cf.monitor_elbo = 100;
cf.fix_first = true;
M = 500;
Q = 2;
test_ind{1} = x >= 10.2 & x <= 10.8;
test_ind{2} = x >= 13.5 & x <= 14.2;
outputs = [2,3];

% a good initialization for hyperparameters
% par.g{1}.loghyp = log([0.5; 1]);
% par.g{2}.loghyp = log(ones(size(par.g{2}.loghyp)));

%% single run 
% par.g = cell(Q,1);
% [elbo,par] = slfm2_learn(x,y,M,par,cf);
% [mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,x,size(y,2));
% mu =mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
% 
% disp('elbo = ')
% disp(elbo(end))
% disp('learned w = ')
% disp(par.w)
% 
% figure;
% %plot(1:numel(elbo),elbo);
% elbo = elbo(elbo ~= 0);
% semilogy(1:numel(elbo),-elbo);
% ylabel('elbo')
% xlabel('iteration')
% title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])
% 
% titles = {'Cambermet','Chimet'};
% smses = zeros(2,1);
% for i=1:2
%   t = outputs(i);
%   plot_all(x,y0(:,t),x,mu(:,t),(ystd(t).^2)*var(:,t),[],[],titles{i});
%   plot(x(test_ind{i}),y0(test_ind{i},t),'.b','MarkerSize',14); % override the magenta color for missing data
%   axis([10 15 10 28]);
%   saveas(gcf, ['results/figures/slfm-weather' titles{i} '.eps'],'epsc');
%   tmp = ~isnan(ytest(:,t)) & test_ind{i};
%   smses(i) = mysmse(ytest(tmp,t),mu(tmp,t),ymean(t));
% end  
% disp(smses)
% disp('average = ')
% disp(mean(smses))

%% batch
runs = 10;
smses = zeros(runs,2);
nlpds = smses;
for r=1:runs
  par.g = cell(Q,1);
  [elbo,par] = slfm2_learn(x,y,M,par,cf);
  [mu,vaar,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,x,size(y,2));
  mu = mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
  fvar = vaar.*repmat(ystd.^2,size(mu,1),1);
  for i=1:2
    t = outputs(i);
    tmp = ~isnan(ytest(:,t)) & test_ind{i};
    smses(r,i) = mysmse(ytest(tmp,t),mu(tmp,t),ymean(t));
    nlpds(r,i) = mynlpd(ytest(tmp,t),mu(tmp,t),fvar(tmp,t));
  end  
end
save('weather-slfm-batch1000-M500','smses','nlpds');
disp('mean smses')
disp(mean(smses(:)))

%% independent gps
% [x,y,xtest,ytest,y0] = read_weather();
% y = y(:,2:3);
% ytest = ytest(:,2:3);
% y0 = y0(:,2:3);
% [y,ymean,ystd] = standardize(y,[],[]);
% mu = size(ytest);
% titles = {'Cambermet','Chimet'};
% runs = 1;
% for run=1:runs
%   for i=1:2
%     validi = ~isnan(y(:,i));
%     model = standard_gp([],x(validi),y(validi,i),xtest,[],0);
%     mu = model.fmean*ystd(i) + ymean(i);
%     var = model.fvar;
%     plot_all(x,y0(:,i),x,mu,(ystd(i).^2)*var,[],[],titles{i});
%     plot(x(test_ind{i}),y0(test_ind{i},i),'.b','MarkerSize',14); % override the magenta color for missing data
%     axis([10 15 10 28]);
%     saveas(gcf, ['results/figures/weather' titles{i} '.eps'],'epsc');
%     tmp = ~isnan(ytest(:,i)) & test_ind{i};
%     smses(run,i) = mysmse(ytest(tmp,i),mu(tmp),ymean(i));
%   end  
% end
% disp(smses)
% disp('average = ')
% disp(mean(smses))
