%% single output (independent gps)
clear all; clc; %close all;
rng(1110,'twister');

[x,y,xtest,ytest] = read_juraCd();
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
title('MAE by SPGP (averaged over 10 runs)');
saveas(gcf, 'jura-fitc.eps','epsc');
save('jura-fitc.mat','maes');

%%
% gpsvi
covfunc   = 'covSEard';
cf.covfunc = covfunc;
cf.lrate     = 1e-2;
cf.lrate_hyp = 1e-4;
cf.lrate_beta = 1e-4;
cf.lrate_z   = 1e-4;
cf.momentum  = 0.9;
cf.momentum_z = 0.0;
cf.learn_z   = true;
cf.init_kmeans = false;
cf.maxiter = 1000;
cf.nbatch = 20;

[mu,s2,elbo,params] = svi_learn(x,y,xtest,M,cf,z0);
disp('svigp fitc mae = ')
disp(mean(abs(mu-ytest)))
%plot_all(x,y,xtest,mu,s2,params.z0,params.z,['GPSVI ' func ' iter = 100']);
%axis(theaxis);
%saveas(gcf, ['results/figures/toy-iter' num2str(cf.maxiter) '-nbatch' num2str(cf.nbatch) '-run2.eps'],'epsc');

figure;
semilogy(1:numel(elbo),elbo);
ylabel('elbo')
xlabel('iteration')
title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])
%saveas(gcf, ['results/figures/' func '-svi-lrate' num2str(cf.lrate_z) '-bound.eps'],'epsc');
disp('elbo = ')
disp(elbo(end))
