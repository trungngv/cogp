clear all; clc;
rng(1110,'twister');

%% General settings
covfunc   = 'covSEard';
PTRAIN    = 0.8; % Proportion of data-points for training
PIND      = 0.2; % Proportion inducing points
KMEANS    = 0;   % Use K-means for inducing locations
SIGMA2N   = 1e-2;
MAXITER   = 100;
JITTER    = 1e-7;
BETA      = 1/0.01; % init for learning, not the true BETA

cf.tol       = 1e-3;
cf.lrate     = 0.01;
cf.lrate_hyp = 1e-5;
cf.momentum = 0.9;
cf.nbatch    = 5; % batch size 
cf.covfunc = covfunc;
cf.beta   = BETA;

%% Generate samples from a GP
x         = linspace(-10,10)';
[n, D]    = size(x);
nhyper    = eval(feval(covfunc));
loghyp  = log(ones(nhyper,1));
cf.loghyp = log(ones(D+1,1)); % random initial covariance hyperparamter
% K = feval(covfunc, loghyp, x);
% y = jit_chol(K)'*randn(n, 1) + sqrt(SIGMA2N)*randn(n, 1);

linfunc = @(x) sin(x) + cos(x) + JITTER;
% x         = linspace(-10,10)';
% D         = size(x,2);
% nhyper    = eval(feval(covfunc));
% loghyper  = log(ones(nhyper,1));
% cf.loghyp  = loghyper;
% %f         = sample_gp(x, covfunc, loghyper, cf.jitter);
f	        = linfunc(x);
y         = f + sqrt(SIGMA2N)*randn(size(f)); % adds isotropic gaussian noise

%% Training, testing
Nall      = size(x,1);
N         = ceil(PTRAIN*Nall);
idx       = randperm(Nall);
idx_train = idx(1:N);
idx_test  = idx(N+1:Nall);
xtrain    = x(idx_train,:); ytrain = y(idx_train);
xtest     = x(idx_test,:);  ytest  = y(idx_test);

%% Inducing point locations
M     = ceil(PIND*N);
idx_z = randperm(N);
idx_z = idx_z(1:M);
z   = xtrain(idx_z,:);
if (KMEANS)
    z   = kmeans(z, xtrain, foptions());
end

%% Learn q(u)
idx = randperm(N);
m   =  y(idx(1:M));
Sinv = 0.01*(1/var(y))*eye(M);
S = inv(Sinv);
l3 = zeros(numel(MAXITER),1);
for i = 1 : MAXITER
  idx = randperm(N, cf.nbatch)';
  [m,S,cf] = sviUpdate(xtrain(idx,:),ytrain(idx),z,m,S,[],[],cf);
  % compute bound
  l3(i) = sviELBO(xtrain,ytrain,z,m,S,cf,[],[],[],[]);
end
figure;
semilogy(1:numel(l3),l3);

Kmm = feval(cf.covfunc, cf.loghyp, z);
Lmm = jit_chol(Kmm,3);
Kmminv = invChol(Lmm);
[mupred varpred] = predict_gpsvi(Kmminv, covfunc, cf.loghyp, m, S, z, x);

%% Plot prediction
figure; hold on;
plotMeanAndStd(x,mupred,sqrt(varpred),[7 7 7]/7.5);
plot(x,y,'r'); hold on;
plot(xtrain, ytrain, 'ro', 'MarkerSize', 8); hold on;
plot(z, min(y)*ones(size(z,1), 1), 'kx', 'MarkerSize', 10);
ylabel('Predictive distribution');

disp('initial hyp .vs learned hyp (including sigma2n)')
fprintf('%.5f\t%.5f\n', [[loghyp;1/BETA],[cf.loghyp;1/cf.beta]]');

fprintf('elbo = %.4f\n', l3(end));

