function [elbo,dloghyp,dw,dz] = ssvi_elbo(x,y,params,cf,A,Knm,Kmminv,Lmm)
%SSVI_ELBO [elbo,dloghyp,dw,dz] = ssvi_elbo(x,y,params,cf,A,Knm,Kmminv,Lmm)
%   
% Lowerbound as a function of the hyperparameters (and inducing inputs)
% for STRUCTURED (multiple-output) gps. Gradients are wrt to the shared
% function g(x).
%
% Usage:
%   elbo = ssvi_elbo()
%   [elbo,dloghyp,dw] = ssvi_elbo()
%   [elbo,~,~,dz] = ssvi_elbo()
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - params : parameters structure
%   - cf : cf.covfunc 
%   - A,Knm,Kmminv,Lmm: saved computation (covariance function of g)
%
% OUTPUT
%   - elbo, dloghyp, dw, dz : objective value and its derivatives
%
% Trung Nguyen
% 21/02/14

z_g = params.g.z;
m_g = params.g.m;
S_g = params.g.S;
P = size(y,2); % # outputs
loghyp_g = params.g.loghyp;
% contribution from g
if nargin == 4 || isempty(A)
  [A,Knm,Kmminv,Lmm] = computeKnmKmminv(cf.covfunc_g, loghyp_g, x, z_g);
end  
diagKnn = feval(cf.covfunc_g, loghyp_g, x, 'diag');
lkl = 0.5*(logdetChol(Lmm) - logdet(S_g) + traceABsym(Kmminv,S_g) + m_g'*Kmminv*m_g); % kl part
ltilde = zeros(P,1);
ltrace = ltilde;
elbo = - lkl;
for i=1:P
  indice = params.idx(:,i);
  betaval = params.task{i}.beta;
  xi = x(indice,:);
  w = params.w(i); w2 = w*w;
  y_minus_g = y(indice,i) - w*A(indice,:)*m_g;
  elbo = elbo + svi_elbo(xi,y_minus_g,params.task{i},cf.covfunc_h,[],[],[],[]);
  ltilde(i) = 0.5*betaval*w2*sum(diagKnn(indice) - diagProd(A(indice,:),Knm(indice,:)')); % Ktilde part
  ltrace(i) = 0.5*betaval*w2*traceABsym(S_g,A(indice,:)'*A(indice,:)); % trace(SA'A) part
  elbo = elbo - ltilde(i) - ltrace(i);
end

dloghyp = []; dw = [];
if nargout >= 2 && nargout < 4  % derivatives 
  dloghyp = zeros(size(loghyp_g));
  dw = zeros(P,1);
  for i=1:P
    indice = params.idx(:,i);
    xi = x(indice,:);
    Ai = computeKnmKmminv(cf.covfunc_h, params.task{i}.loghyp, xi, params.task{i}.z);
    y_minus_hi = y(params.idx(:,i),i) - Ai*params.task{i}.m;
    betaval = params.task{i}.beta;
    params.g.beta = betaval;
    params.g.w = params.w(i);
    [~,dloghyp_i] = ssvi_elbo_g(xi,y_minus_hi,params.g,cf.covfunc_g,P);
    dloghyp = dloghyp + dloghyp_i;
    w = params.w(i);
    Am = A(indice,:)*m_g;
    dw(i) = -(2/w)*ltilde(i) -(2/w)*ltrace(i) + betaval*y_minus_hi'*Am - betaval*w*sum(Am.^2);
  end
end

if nargout == 4
  dz = zeros(size(z_g));
  for i=1:P
    indice = params.idx(:,i);
    xi = x(indice,:);
    Ai = computeKnmKmminv(cf.covfunc_h, params.task{i}.loghyp, xi, params.task{i}.z);
    y_minus_hi = y(params.idx(:,i),i) - Ai*params.task{i}.m;
    betaval = params.task{i}.beta;
    params.g.beta = betaval;
    params.g.w = params.w(i);
    [~,~,dz_i] = ssvi_elbo_g(xi,y_minus_hi,params.g,cf.covfunc_g,P);
    dz = dz + dz_i;
  end
end
% dm,dS (only for gradient checking)
% if nargout == 5
%   Lambda = Kmminv;
%   tmp = zeros(size(params.g.m));
%   for i=1:P
%     w = params.w(i); w2 = w*w;
%     betaval = params.task{i}.beta;
%     indice = params.idx(:,i);
%     Lambda = Lambda + betaval*w2*(A(indice,:)')*A(indice,:);
%     Ai = computeKnmKmminv(cf.covfunc_h, params.task{i}.loghyp, x(indice,:), params.task{i}.z);
%     y_minus_hi = y(indice,i) - Ai*params.task{i}.m;
%     tmp = tmp + betaval*w*A(indice,:)'*y_minus_hi;
%   end
% 
%   Sinv = invChol(jit_chol(params.g.S,4));
%   dm = tmp - Lambda*params.g.m;
%   dS = 0.5*Sinv - 0.5*Lambda;
% end

end

