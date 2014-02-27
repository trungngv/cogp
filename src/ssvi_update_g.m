function [params,A,Knm,Kmminv,Lmm] = ssvi_update_g(x,y,params,cf)
%SSVI_UPDATE_G [params,A,Knm,Kmminv,Lmm] = ssvi_update_g(x,y,params,cf)
%   
% Update the variational parameters of the shared process g in a structured
% gp model.
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - params
%   - cf : options for the optimization procedure
%
% OUTPUT
%   - params : updated parameters
%   - A,Knm,Kmminv,Lmm : saved computation (covariance of g)
%
% Trung V. Nguyen
% 20/02/14
P = size(y,2);
[A,Knm,Kmminv,Lmm] = computeKnmKmminv(cf.covfunc_g, params.g.loghyp, x, params.g.z);
Lambda = Kmminv;
tmp = zeros(size(params.g.m));
for i=1:P
  w = params.w(i); w2 = w*w;
  betaval = params.task{i}.beta;
  indice = params.idx(:,i);
  Lambda = Lambda + betaval*w2*(A(indice,:)')*A(indice,:);
  Ai = computeKnmKmminv(cf.covfunc_h, params.task{i}.loghyp, x(indice,:), params.task{i}.z);
  y_minus_hi = y(indice,i) - Ai*params.task{i}.m;
  tmp = tmp + betaval*w*A(indice,:)'*y_minus_hi;
end

Sinv = invChol(jit_chol(params.g.S));
theta1_old = Sinv*params.g.m;
theta2_old = -0.5*Sinv;
theta2 = theta2_old + cf.lrate*(-0.5*Lambda - theta2_old);
theta1 = theta1_old + cf.lrate*(tmp - theta1_old);
[VV DD] = eig(theta2);
invTheta2 = VV*diag(1./diag(DD))*VV';
params.g.S = - 0.5*invTheta2;
params.g.m = params.g.S*theta1;

end

