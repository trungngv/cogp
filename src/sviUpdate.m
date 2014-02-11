function [m,S,cf] = sviUpdate(x,y,z,m,S,Lmm,Kmminv,cf)
%SVIUPDATE [m,S] = sviUpdate(x,y,z,m,S,Lmm,Kmminv,cf)
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
%   - z : inducing inputs
%   - m,S : current variational parameters
%   - Lmm : [optional] chol(Kmm)
%   - Kmminv : [optional] Kmm^{-1} (must update if hyper is not fixed)
%   - cf : options for the optimization procedure and hyperparameters
%
% OUTPUT
%   - m, S : mean and covariance of q(u)
%   - cf : loghyp and beta updated
%
% TODO: add momentum term for loghyp
N = size(x,1);
M = size(z,1);
Sinv = invChol(jit_chol(S,4));
theta1_old = Sinv*m;
theta2_old = -0.5*Sinv;
betaval = cf.beta;
if isempty(Kmminv)
  Kmm = feval(cf.covfunc, cf.loghyp, z);
  Lmm = jit_chol(Kmm,4); clear Kmm;
  Kmminv = invChol(Lmm);
end

Knm = feval(cf.covfunc, cf.loghyp, x, z);
A = Knm*Kmminv;
Lambda = betaval*(A'*A) + Kmminv;

% Stochastic natural ascent: VB-E update for m,S
theta2 = theta2_old + cf.lrate*(-0.5*Lambda - theta2_old); clear Lambda;
theta1 = theta1_old + cf.lrate*(betaval*A'*y - theta1_old);

%TODO replace with jit_chol and invChol
[VV DD]     = eig(theta2);
invTheta2 = VV*diag(1./diag(DD))*VV';
logOld = sviELBO(x,y,z,m,S,cf,[],[],[],[]);
S           = - 0.5*invTheta2;
m           = S*theta1;

% VB-M step for hyperparameters and perform gradient check
% Note covSEard returns dK / dloghyper 
[logL,dloghyp,dbeta] = sviELBO(x,y,z,m,S,cf,Lmm,Kmminv,Knm,A);
cf.loghyp = cf.loghyp + cf.lrate_hyp*dloghyp;
% faster learn rate for the noise 
cf.beta = cf.beta + cf.lrate_hyp*dbeta;
logNew = sviELBO(x,y,z,m,S,cf,[],[],[],[]);
assert(logOld <= logL && logL <= logNew);

return;

