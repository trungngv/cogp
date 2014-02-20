function params = svi_update(x,y,params,cf,covfunc,Lmm,Kmminv)
%SVI_UPDATE params = svi_update(x,y,params,cf,covfunc,Lmm,Kmminv)
%   
% Update the variational parameters (of the posterior q(u|y) = N(u; m, S))
% and the covariance hyperparameters of a standard GP regression using SVI.
% This update is composed of a VB-update (of the variational parameter using
% natural gradient) and a standard VB-update of the covaraince
% hyperparamters.
%
% If x is the entire training, this becomes the (natural) variational
% inference. 
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
%   - Lmm : [optional] chol(Kmm)
%   - Kmminv : [optional] Kmm^{-1} (must update if hyper is not fixed)
%
% OUTPUT
%   - params : updated parameters
%
% Trung V. Nguyen
% 17/02/14
Sinv = invChol(jit_chol(params.S,4));
theta1_old = Sinv*params.m;
theta2_old = -0.5*Sinv;
betaval = params.beta;
if isempty(Kmminv)
  Kmm = feval(covfunc, params.loghyp, params.z);
  Lmm = jit_chol(Kmm,4); clear Kmm;
  Kmminv = invChol(Lmm);
end

Knm = feval(covfunc, params.loghyp, x, params.z);
A = Knm*Kmminv;
Lambda = betaval*(A'*A) + Kmminv;

% Stochastic natural ascent: VB-E update for m,S
theta2 = theta2_old + cf.lrate*(-0.5*Lambda - theta2_old);
theta1 = theta1_old + cf.lrate*(betaval*A'*y - theta1_old);
[VV DD] = eig(theta2);
invTheta2 = VV*diag(1./diag(DD))*VV';
params.S = - 0.5*invTheta2;
params.m = params.S*theta1;

% VB-M step for hyperparameters and perform gradient check
% Note covSEard returns dK / dloghyper 
if cf.learn_z
  [logL,dloghyp,dbeta,dz] = svi_elbo(x,y,params,covfunc,Lmm,Kmminv,Knm,A);
  params = stochastic_update(params,cf,dloghyp,dbeta,dz);
else
  [logL,dloghyp,dbeta] = svi_elbo(x,y,params,covfunc,Lmm,Kmminv,Knm,A);
  params = stochastic_update(params,cf,dloghyp,dbeta,[]);
end

return;

