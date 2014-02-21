function [params,A,Knm,Kmminv,Lmm] = svi_update(x,y,params,cf,covfunc)
%SVI_UPDATE [params,A,Knm,Kmminv,Lmm] = svi_update(x,y,params,cf,covfunc)
%   
% Update the variational parameters (of the posterior q(u|y) = N(u; m, S)).
% 
% This update corresponds to the E-step in a variational EM procedure.
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
%   - covfunc: covariance function
%
% OUTPUT
%   - params : updated parameters
%   - A,Knm,Kmminv,Lmm: saved computation
%
% Trung V. Nguyen
% 20/02/14
betaval = params.beta;
[A,Knm,Kmminv,Lmm] = computeKnmKmminv(covfunc, params.loghyp, x, params.z);
Lambda = betaval*(A'*A) + Kmminv;
Sinv = invChol(jit_chol(params.S,4));
theta1_old = Sinv*params.m;
theta2_old = -0.5*Sinv;
theta2 = theta2_old + cf.lrate*(-0.5*Lambda - theta2_old);
theta1 = theta1_old + cf.lrate*(betaval*A'*y - theta1_old);
[VV DD] = eig(theta2);
invTheta2 = VV*diag(1./diag(DD))*VV';
params.S = - 0.5*invTheta2;
params.m = params.S*theta1;

return;

