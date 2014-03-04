%% single output (independent gps)
clear all; clc; %close all;
rng(1110,'twister');

[x,y,xtest,ytest] = load_data('data/jura','juraCd');
x = x(1:259,:); y = y(1:259,1);
ytest = ytest(:,1);
y = log(y);
[y,ymean,ystd] = standardize(y,[],[]);

Ms = [5,10,20,50,100,150,200,259];
%% fitc
runs = 10;
maes = zeros(numel(Ms),runs);
for i=1:numel(Ms)
  M = Ms(i);
  for j=1:runs
    z0 = x(randperm(size(x,1),M),:);
    [z,z0,~,mu,s2] = learn_fitc(x,y,xtest,M,z0);
    mu = exp(mu*ystd + ymean);
    maes(i,j) = mean(abs(ytest - mu));
  end
end

mae_mean = mean(maes,2);
mae_std = 2*std(maes,0,2);
plot(Ms,mae_mean,'x-','MarkerSize',15);
errorbar(Ms,mae_mean,mae_std);
xlabel('number of inducing points');
ylabel('mae');
title('fitc, mae vs. inducing inputs');
saveas(gcf, 'juraCu-fitc.eps','epsc');
save('juraCu-fitc.mat','maes');

%% standard gp
model = standard_gp([],x,y,xtest,[],false);
mu = exp(model.fmean * ystd + ymean);
disp('mae = ')
disp(mean(abs(mu - ytest)));
%%
% gpsvi
rng(1110,'twister');
covfunc   = 'covSEard';
cf.covfunc = covfunc;
cf.lrate     = 1e-2;
cf.lrate_hyp = 1e-5;
cf.lrate_beta = 1e-4;
cf.lrate_z   = 1e-6;
cf.momentum  = 0.9;
cf.momentum_z = 0.9;
cf.learn_z   = true;
cf.init_kmeans = true;
cf.maxiter = 1000;
cf.nbatch = 50;

M = 150;
z0 = x(randperm(size(x,1),M),:);
[mu,s2,elbo,params] = svi_learn(x,y,xtest,M,cf,[]);
mu = exp(mu*ystd + ymean);
disp('mae = ')
disp(mean(abs(ytest - mu)));

% batch 
% runs = 10;
% maes = zeros(numel(Ms),runs);
% for i=1:numel(Ms)
%   M = Ms(i);
%   for j=1:runs
%     z0 = x(randperm(size(x,1),M),:);
%     [mu,s2,elbo,params] = svi_learn(x,y,xtest,M,cf,z0);
%     mu = exp(mu*ystd + ymean);
%     maes(i,j) = mean(abs(ytest - mu));
%     disp('elbo = ')
%     disp(elbo(end))
%   end
% end
% 
% mae_mean = mean(maes,2);
% mae_std = 2*std(maes,0,2);
% plot(Ms,mae_mean,'x-','MarkerSize',15);
% errorbar(Ms,mae_mean,mae_std);
% xlabel('number of inducing points');
% ylabel('mae');
% title('svigp optimize z, mae vs. inducing inputs');
% saveas(gcf, 'juraCu-svigp-batch50.eps','epsc');
% save('juraCu-svigp-batch50.mat','maes');

figure;
plot(1:numel(elbo),elbo);
ylabel('elbo')
xlabel('iteration')
title(['elbo vs. iteration'])
disp('elbo = ')
disp(elbo(end))
%saveas(gcf, ['results/figures/' func '-svi-lrate' num2str(cf.lrate_z) '-bound.eps'],'epsc');

