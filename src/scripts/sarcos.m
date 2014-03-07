clear all; clc;
load sarcos_inv
load sarcos_inv_test
fmae = @(ytrue,ypred) mean(abs(ypred-ytrue(:,1)));
frmse = @(ytrue,ypred) sqrt(mean((ypred-ytrue(:,1)).^2));

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
clear sarcos_inv
clear sarcos_inv_test

%% Exploratory analysis of sarcos
% correlation = zeros(7,7);
% for i=1:7
%   for j=1:7
%     correlation(i,j) = corr(y(:,i),y(:,j));
%   end
% end
% 
% disp('correlation: ')
% correlation
% 
% outputs = [2,3,4,7];
% for i=1:numel(outputs)-1
%   for j=(i+1):numel(outputs)
%     figure; scatter(y(:,i),y(:,j)); title(['output ' num2str(i) ' and ' num2str(j)']);
%   end
% end

%% independent models - fitc
% rng(1110,'twister');
% M = 200;
% runs = 5;
% maes = zeros(runs,size(y,2));
% rmses = maes;
% smses = maes;
% for n=1:runs
%   for i=1:size(y,2)
%     [z,z0,~,mu,s2] = learn_fitc(x,y(:,i),xtest,M,[]);
%     mu = mu*ystd(i)+ymean(i);
%     disp('mae    rmse')
%     fprintf('%.4f   %.4f\n', fmae(ytest(:,i),mu), frmse(ytest(:,i),mu));
%     maes(n,i) = fmae(ytest(:,i),mu);
%     rmses(n,i) = frmse(ytest(:,i),mu);
%     smses(n,i) = mysmse(ytest(:,i),mu,ymean(i));
%   end  
% end  
% save(['sarcos-fitc-M' num2str(M)],'maes','rmses','smses');
% disp('average mae, rmse, smse')
% disp(mean(maes))
% disp(mean(rmses))
% disp(mean(smses))

%% independent models - svigp
rng(1110,'twister');
covfunc   = 'covSEard';
cf.covfunc = covfunc;
cf.lrate     = 1e-2;
cf.lrate_hyp = 1e-5;
cf.lrate_beta = 1e-4;
cf.lrate_z   = 1e-4;
cf.momentum  = 0.9;
cf.momentum_z = 0.0;
cf.learn_z   = true;
cf.maxiter = 1000;
cf.nbatch = 5000;
cf.beta = 1/0.01;
cf.initz = 'random';
cf.monitor_elbo = 50;

M = 200;
runs = 1;
maes = zeros(runs,size(y,2));
smses = maes;
for n=1:runs
  for i=1:size(y,2)
    [mu,s2,elbo,params] = svi_learn(x,y(:,i),xtest,M,cf,[]);
    mu = mu*ystd(i)+ymean(i);
    disp('mae    smse')
    fprintf('%.4f   %.4f\n', fmae(ytest(:,i),mu), frmse(ytest(:,i),mu));
    maes(n,i) = fmae(ytest(:,i),mu);
    smses(n,i) = mysmse(ytest(:,i),mu,ymean(i));
  end  
end  

disp('average mae and smses')
disp(mean(maes))
disp(mean(smses))

%% independent model - soD
% rng(1110,'twister');
% runs = 5;
% maes = zeros(runs,size(y,2));
% rmses = maes;
% smses = maes;
% for n=1:runs
%   for i=1:size(y,2)
%     randind = randperm(size(x,1),2000);
%     xsod = x(randind,:);
%     ysod = y(randind,i);
%     model = standard_gp([],xsod,ysod,xtest,[],false);
%     mu = model.fmean*ystd(i)+ymean(i);
%     disp('mae    rmse')
%     fprintf('%.4f   %.4f\n', fmae(ytest(:,i),mu), frmse(ytest(:,i),mu));
%     maes(n,i) = fmae(ytest(:,i),mu);
%     rmses(n,i) = frmse(ytest(:,i),mu);
%     smses(n,i) = mysmse(ytest(:,i),mu,ymean(i));
%   end  
% end  
% save(['sarcos-sod'],'maes','rmses','smses','model');
% disp('average mae, rmse, smse')
% disp(mean(maes))
% disp(mean(rmses))
% disp(mean(smses))
