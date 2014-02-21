function params = svi_update_hyper(x,y,params,cf,covfunc,A,Knm,Kmminv,Lmm)
%SVI_UPDATE_HYPER params = svi_update_hyper(x,y,params,cf,covfunc,A,Knm,Kmminv,Lmm)
%   
% Update the hyperparameters (including the inducing inputs) of a standard
% GP regression using stochastic variational inference.
% This update is the M-step in the variational EM procedure.
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - params : structure containing variational parameters (m,S), inducing
%   inputs, and hyperparameters
%       params.m, params.S : variatinal parameters of the posterior
%       params.z : inducing inputs
%       params.loghyp : cov hyperparameters
%       params.beta : noise hyperparameter
%       params.delta_z,delta_beta,delta_loghyp: for SGD learning
%   - cf : options for the optimization procedure
%   - covfunc: 
%   - Lmm,Kmminv,Knm,A : saved computation
%
% OUTPUT
%   - params : updated parameters
%
% Trung V. Nguyen
% 20/02/14
if cf.learn_z
  [logL,dloghyp,dbeta,dz] = svi_elbo(x,y,params,covfunc,A,Knm,Kmminv,Lmm);
  params = stochastic_update(params,cf,dloghyp,dbeta,dz);
else
  [logL,dloghyp,dbeta] = svi_elbo(x,y,params,covfunc,A,Knm,Kmminv,Lmm);
  params = stochastic_update(params,cf,dloghyp,dbeta,[]);
end

