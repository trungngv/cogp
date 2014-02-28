function [mu,s2,elbo,params] = svi_learn(x,y,xtest,M,cf,z0)
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
params.beta = 1/0.01;
params.delta_beta = 0;
elbo = zeros(numel(cf.maxiter),1);
for i = 1 : cf.maxiter
  idx = randperm(N, cf.nbatch);
  xi = x(idx,:); yi = y(idx);
  [params,A,Knm,Kmminv,Lmm] = svi_update(xi,yi,params,cf,cf.covfunc);
  if cf.learn_z
    [~,dloghyp,dbeta,dz] = svi_elbo(xi,yi,params,cf.covfunc,A,Knm,Kmminv,Lmm);
    params = stochastic_update(params,cf,dloghyp,dbeta,dz);
  else
    [~,dloghyp,dbeta] = svi_elbo(xi,yi,params,cf.covfunc,A,Knm,Kmminv,Lmm);
    params = stochastic_update(params,cf,dloghyp,dbeta,[]);
  end
  % compute elbo using updated parameters
  elbo(i) = svi_elbo(x,y,params,cf.covfunc,[],[],[],[]);
end

if ~isempty(xtest)
  [mu s2] = gpsvi_predict(cf.covfunc, params.loghyp, params.m, params.S, params.z, xtest);
end

end

