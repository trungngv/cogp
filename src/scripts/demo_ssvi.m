clear all; clc; %close all;
%rng(1111,'twister');

% sin(x) + cos(x)
x = linspace(-10,10)';
N = size(x,1);
xtest = linspace(-11,11)';
% fn1 = @(x) 0.2*x + sin(x) + 1e-7;
% fn2 = @(x) 0.2*x + sin(x) + 1e-7;
fn1 = @(x) sin(x) + 1e-7;
fn2 = @(x) -sin(x) + 1e-7;
y1 = fn1(x) + sqrt(1e-4)*randn(N,1);
y2 = fn2(x) + sqrt(1e-4)*randn(N,1);

ind1 = -7 < x & x < -3;
ind2 = 4 < x & x < 8;
y1miss = y1;
y1miss(ind1) = nan;
y2miss = y2;
y2miss(ind2) = nan;
y = [y1miss, y2miss];
%y = y1;
M.g = 20; M.h = 20;

% ssvi
cf.covfunc_g  = 'covSEard';
cf.covfunc_h  = 'covNoise';
cf.lrate      = 1e-2;
cf.momentum   = 0.9;
cf.lrate_hyp  = 1e-5;
cf.lrate_beta = 1e-4;
cf.momentum_w = 0.9;
cf.lrate_w    = 1e-5;
cf.learn_z    = true;
cf.momentum_z = 0.0;
cf.lrate_z    = 1e-4;
cf.init_kmeans = false;
cf.maxiter = 500;
cf.nbatch = 10;

[elbo,params] = ssvi_learn(x,y,M,cf,[]);
[mu,var,mu_g,var_g] = ssvi_predict(cf.covfunc_g,cf.covfunc_h,params,xtest);

plot_all(x,y1,xtest,mu(:,1),var(:,1),[],params.task{1}.z,'y1');
axis([-11 11 -4 4]);
plot_all(x,y2,xtest,mu(:,2),var(:,2),[],params.task{2}.z,'y2');
axis([-11 11 -4 4]);
plot_all(x,y1,xtest,mu_g(:,1),var_g(:,1),params.g.z0,params.g.z,'y1 by g');
axis([-11 11 -4 4]);

figure;
plot(1:numel(elbo),elbo);
ylabel('elbo')
xlabel('iteration')
title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])

disp('elbo = ')
disp(elbo(end))
disp('learned w = ')
disp(params.w)

%% independent gpsvi
% cf.covfunc = 'covSEard';
% cf.lrate_z = 1e-3;
% x1 = x(~ind1);
% [mu,s2,elbo,params] = svi_learn(x1,y1(~ind1),xtest,M,cf,[]);
% plot_all(x,y1,xtest,mu,s2,params.z0,params.z,'independent y1');
% axis([-11 11 -4 4]);
% 
% x2 = x(~ind2);
% [mu,s2,elbo,params] = svi_learn(x2,y2(~ind2),xtest,M,cf,[]);
% plot_all(x,y2,xtest,mu,s2,params.z0,params.z,'independent y2');
% axis([-11 11 -4 4]);
