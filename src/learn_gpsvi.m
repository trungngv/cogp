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
nhyper = eval(feval(cf.covfunc));
params = init_params(x,y,M,nhyper,0,z0);
elbo = zeros(numel(cf.maxiter),1);
for i = 1 : cf.maxiter
  idx = randperm(N, cf.nbatch);
  params = svi_update(x(idx,:),y(idx),params,cf,cf.covfunc,[],[]);
  elbo(i) = svi_elbo(x,y,params,cf.covfunc,[],[],[],[]);
end

if ~isempty(xtest)
  [mu s2] = gpsvi_predict(cf.covfunc, params.loghyp, params.m, params.S, params.z, xtest);
end

end

