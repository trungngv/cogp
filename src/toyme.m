clear all; clc; close all;
rng(1111,'twister');

CHOICE = 1;
switch CHOICE
  case 1
    % The function in SPGP 
    x = load('train_inputs');
    y = load('train_outputs');
    xtest = load('test_inputs');
    func = 'func1';
    theaxis = [-4 10 -2.5 1.5];
  case 2
    % sin(x) + cos(x)
    x = linspace(-10,10)';
    xtest = linspace(-11,11)';
    linfunc = @(x) sin(x) + cos(x) + 1e-7;
    f	= linfunc(x);
    y = f + sqrt(1e-2)*randn(size(f)); % adds isotropic gaussian noise
    func = 'func2';
    theaxis = [-11 11 -2.5 2.5];
  case 3
    % sample from a gp
    x = linspace(-10,10)';
    xtest = linspace(-11,11)';
    f = sample_gp(x, 'covSEard', log([2;1]), 1);
    y = f + sqrt(1e-2)*randn(size(f));
    func = 'func3';
    theaxis = [-11 11 -2.5 2.5];
end

M = 10;
z0 = x(randperm(size(x,1),M),:);

% fitc
[z,z0,~,mu,s2] = learn_fitc(x,y,xtest,M,z0);
plot_all(x,y,xtest,mu,s2,z0,z,'FITC');
axis(theaxis);
%saveas(gcf, ['results/figures/' func '-fitc.eps'],'epsc');

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
cf.maxiter = 500;
cf.nbatch = 5;

[mu,s2,elbo,params] = learn_gpsvi(x,y,xtest,M,cf,z0);
plot_all(x,y,xtest,mu,s2,params.z0,params.z,['GPSVI lrate = ' num2str(cf.lrate_z)]);
axis(theaxis);
%saveas(gcf, ['results/figures/' func '-svi-lrate' num2str(cf.lrate_z) '.eps'],'epsc');

figure;
semilogy(1:numel(elbo),elbo);
ylabel('elbo')
xlabel('iteration')
title(['elbo vs. iteration, lrate = ' num2str(cf.lrate_z)])
%saveas(gcf, ['results/figures/' func '-svi-lrate' num2str(cf.lrate_z) '-bound.eps'],'epsc');
disp('elbo = ')
disp(elbo(end))
