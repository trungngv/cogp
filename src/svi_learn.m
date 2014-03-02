function [mu,s2,elbo,par] = svi_learn(x,y,xtest,M,cf,z0)
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
par = init_params(x,y,M,nhyper,0,z0);
par.beta = 1/0.05;
par.delta_beta = 0;
elbo = zeros(numel(cf.maxiter),1);
for i = 1 : cf.maxiter
  idx = randperm(N, cf.nbatch);
  xi = x(idx,:); yi = y(idx);
  [par,A,Knm,Kmminv,Lmm] = svi_update(xi,yi,par,cf,cf.covfunc);
  [~,dloghyp,dbeta] = svi_elbo(xi,yi,par,cf.covfunc,A,Knm,Kmminv,Lmm);
  [par.loghyp,par.delta_hyp] = stochastic_update(par.loghyp,par.delta_hyp,...
    dloghyp, cf.momentum, cf.lrate_hyp);
  [par.beta,par.delta_beta] = stochastic_update(par.beta,par.delta_beta,...
    dbeta, cf.momentum, cf.lrate_beta);
  if cf.learn_z
    [~,~,~,dz] = svi_elbo(xi,yi,par,cf.covfunc);
    [par.z,par.delta_z] = stochastic_update(par.z, par.delta_z, dz, ...
      cf.momentum_z, cf.lrate_z);
  end
  % compute elbo using updated parameters
  elbo(i) = svi_elbo(x,y,par,cf.covfunc,[],[],[],[]);
end

if ~isempty(xtest)
  [mu s2] = gpsvi_predict(cf.covfunc, par.loghyp, par.m, par.S, par.z, xtest);
end

end

