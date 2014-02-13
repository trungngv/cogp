function [mu,s2,elbo,params] = learn_gpsvi(x,y,xtest,M,cf,z0)
%LEARN_GPSVI learn_gpsvi(x,y,xtest,M,cf,z0)
%   Quick wrapper to run GPSVI.
% 
% INPUT
%   - x,y,xtest : train inputs, train outputs, test outputs
%   - M : #inducing inputs
%   - cf : configuration
%
% OUTPUT
%   - mu, s2 : mean and variance prediction
%   - elbo : the bound
%   - params : learned parameters
[N D] = size(x);
idx_z = randperm(N,M);
if isempty(z0)
  z0 = x(idx_z,:);
  if cf.init_kmeans
    z0 = kmeans(z0, x, foptions());
  end
end
idx = randperm(N,M);
m =  y(idx);  % shouldn't this be the y of idx_z?
%m = y(idx_z);
Sinv = 0.01*(1/var(y))*eye(M);
S = inv(Sinv);
params.m = m;
params.S = S;
params.z = z0;
params.z0 = z0;
nhyper = eval(feval(cf.covfunc));
loghyp = log(ones(nhyper,1));
%sigma2n = var(y-mean(y),1)/10;
beta0 = 1/0.01;
params.beta   = beta0; % should be init using the data
params.loghyp = loghyp;
params.delta_hyp = zeros(size(params.loghyp));
params.delta_beta = 0;
params.delta_z = zeros(size(z0));

elbo = zeros(numel(cf.maxiter),1);
for i = 1 : cf.maxiter
  idx = randperm(N, cf.nbatch);
  params = sviUpdate(x(idx,:),y(idx),params,cf,[],[]);
  elbo(i) = sviELBO(x,y,params,cf,[],[],[],[]);
end

Kmm = feval(cf.covfunc, params.loghyp, params.z);
Lmm = jit_chol(Kmm,3);
Kmminv = invChol(Lmm);
if ~isempty(xtest)
  [mu s2] = predict_gpsvi(Kmminv, cf.covfunc, params.loghyp, params.m, params.S, params.z, xtest);
end

disp('initial hyp .vs learned hyp (including sigma2n)')
fprintf('%.5f\t%.5f\n', [[loghyp;1/beta0],[params.loghyp;1/params.beta]]');

end

