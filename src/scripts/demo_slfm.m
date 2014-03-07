clear all; clc; %close all;
rng(1111,'twister');

% sin(x) + cos(x)
x = linspace(-10,10,200)';
N = size(x,1);
xtest = linspace(-11,11)';
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
Mstr.g = 15; Mstr.h = 10;

% ssvi
cf.covfunc_g  = 'covSEard';
cf.covfunc_h  = 'covNoise';
cf.lrate      = 1e-2;
cf.momentum   = 0.9;
cf.lrate_hyp  = 1e-5;
cf.lrate_beta = 1e-5;
cf.momentum_w = 0.9;
cf.lrate_w    = 1e-5;
cf.learn_z    = false;
cf.momentum_z = 0.9;
cf.lrate_z    = 1e-4;
cf.maxiter = 500;
cf.nbatch = 20;
cf.beta = 1/0.01;
cf.initz = 'random';
cf.monitor_elbo = 1;
cf.w = ones(size(y,2),1);
cf.fix_first = true;
M = 5;

Q = 1;
par.task = cell(size(y,2),1);
par.g = cell(Q,1);
[elbo,par] = slfm_learn(x,y,Mstr,par,cf);
[mu,var,mu_g,var_g] = slfm_predict(cf.covfunc_g,cf.covfunc_h,par,xtest);
% [elbo,par] = slfm2_learn(x,y,M,par,cf);
% [mu,var,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,xtest,size(y,2));

plot_all(x,y1,xtest,mu(:,1),var(:,1),par.g{1}.z,[],'output 1');
plot(x(ind1),y1(ind1),'.b','MarkerSize',14); % override the magenta color for missing data
axis([-11 11 -4 4]);
saveas(gcf, ['results/figures/toy-slfm-y1.eps'],'epsc');

plot_all(x,y2,xtest,mu(:,2),var(:,2),par.g{1}.z,[],'output 2');
plot(x(ind2),y2(ind2),'.b','MarkerSize',14); % override the magenta color for missing data
axis([-11 11 -4 4]);
saveas(gcf, ['results/figures/toy-slfm-y2.eps'],'epsc');

figure;
plot(1:numel(elbo),elbo);
ylabel('elbo')
xlabel('iteration')
title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])

disp('elbo = ')
disp(elbo(end))
disp('learned w = ')
disp(par.w)

%% independent gpsvi
cf.covfunc = 'covSEard';
M = 15;
cf.nbatch = 20;
x1 = x(~ind1);
[mu,s2,elbo,params] = svi_learn(x1,y1(~ind1),xtest,M,cf,par.g{1}.z);
plot_all(x,y1,xtest,mu,s2,params.z,[],'output 1');
plot(x(ind1),y1(ind1),'.b','MarkerSize',14); % override the magenta color for missing data
axis([-11 11 -4 4]);
saveas(gcf, ['results/figures/toy-svigp-y1.eps'],'epsc');
% 
x2 = x(~ind2);
[mu,s2,elbo,params] = svi_learn(x2,y2(~ind2),xtest,M,cf,par.g{1}.z);
plot_all(x,y2,xtest,mu,s2,params.z,[],'output 2');
plot(x(ind2),y2(ind2),'.b','MarkerSize',14); % override the magenta color for missing data
axis([-11 11 -4 4]);
saveas(gcf, ['results/figures/toy-svigp-y2.eps'],'epsc');
