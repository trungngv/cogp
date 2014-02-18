function [elbo,dloghyp,dz] = ssvi_elbo(x,y,params,cf)
%SSVI_ELBO [elbo,dloghyp,dz] = ssvi_elbo(x,y,params,cf)
%   
% Lowerbound as a function of the hyperparameters (and inducing inputs)
% for STRUCTURED (multiple-output) gps. Gradients are wrt to the shared
% function g(x).
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - params : parameters structure
%   - cf : cf.covfunc 
%
% OUTPUT
%   - elbo, dloghyp, dz : objective value and its derivatives
%
% Trung Nguyen
% 17/02/14

% TODO:
% 1) add weight w_i
% 2) also output derivatives of independent processes h_i in here 
z_g = params.g.z;
m_g = params.g.m;
S_g = params.g.S;
P = size(y,2); % # outputs
loghyp_g = params.g.loghyp;

% contribution from g
Kmm = feval(cf.covfunc_g, loghyp_g, z_g);
Lmm = jit_chol(Kmm,3);
Kmminv = invChol(Lmm);
Knm = feval(cf.covfunc_g, loghyp_g, x, z_g);
A = Knm*Kmminv;
diagKnn = feval(cf.covfunc_g, loghyp_g, x, 'diag');
lkl = 0.5*(logdetChol(Lmm) - logdet(S_g) + traceABsym(Kmminv,S_g) + m_g'*Kmminv*m_g); % kl part
elbo = - lkl;
for i=1:P
  indice = params.idx(:,i);
  xi = x(indice,:);
  y_minus_g = y(indice,i) - A(indice,:)*m_g;
  elbo = elbo + svi_elbo(xi,y_minus_g,params.task{i},cf.covfunc_h,[],[],[],[]);
  ltilde = 0.5*params.task{i}.beta*sum(diagKnn(indice) - diagProd(A(indice,:),Knm(indice,:)')); % Ktilde part
  ltrace = 0.5*params.task{i}.beta*traceABsym(S_g,A(indice,:)'*A(indice,:)); % trace(SA'A) part
  elbo = elbo - ltilde - ltrace;
end

if nargout >= 2
  dloghyp = zeros(size(loghyp_g));
  dz = zeros(size(z_g));
  for i=1:P
    % TODO: add weight w_i
    xi = x(params.idx(:,i),:);
    Ai = computeKnmKmminv(cf.covfunc_h, params.task{i}.loghyp, xi, params.task{i}.z);
    y_minus_hi = y(params.idx(:,i),i) - Ai*params.task{i}.m;
    params.g.beta = params.task{i}.beta;
    if nargout == 3
      [~,dloghyp_i,dz_i] = ssvi_elbo_g(xi,y_minus_hi,params.g,cf.covfunc_g,P);
      dz = dz + dz_i;
    else
      [~,dloghyp_i] = ssvi_elbo_g(xi,y_minus_hi,params.g,cf.covfunc_g,P);
    end
    dloghyp = dloghyp + dloghyp_i;
  end
  
% dm,dS
%   Sinv = invChol(jit_chol(S_g,4));
%   A = computeKnmKmminv(cf.covfunc_g,params.g.loghyp,x,params.g.z);
%   Lambda = Kmminv;
%   tmp = zeros(size(m_g));
%   for i=1:P
%     betaval = params.task{i}.beta;
%     indice = params.idx(:,i);
%     Lambda = Lambda + betaval*(A(indice,:)')*A(indice,:);
%     Ai = computeKnmKmminv(cf.covfunc_h, params.task{i}.loghyp, x(indice,:), params.task{i}.z);
%     y_minus_hi = y(indice,i) - Ai*params.task{i}.m;
%     tmp = tmp + betaval*A(indice,:)'*y_minus_hi;
%   end
%   dm = tmp - Lambda*m_g;
%   dS = 0.5*Sinv - 0.5*Lambda;

end
end

