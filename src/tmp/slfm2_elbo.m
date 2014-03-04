function [elbo,dbeta,dw,dloghyp,dz,dm,dS] = slfm2_elbo(x,y,par,cf)
%SLFM_ELBO [elbo,dbeta,dw,dloghyp,dz,dm,dS] = slfm2_elbo(x,y,par,cf)
%   
% Lowerbound as a function of the hyperparameters (and inducing inputs)
% for STRUCTURED/MULTIPLE-OUTPUT gps.
%
% Usage:
%   elbo = slfm_elbo()
%   [elbo,dbeta,dw,dloghyp] = slfm_elbo()
%   [elbo,~,~,~,dz] = slfm_elbo()
%   [elbo,~,~,~,~,dm,dS] = slfm_elbo()
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - par : parameters structure
%   - cf : cf.covfunc 
%
% OUTPUT
%   - elbo, dbeta, dloghyp, dw, dz : objective value and its derivatives
%
% Trung Nguyen
% 27/02/14
P = size(y,2); Q = numel(par.g);
elbo = 0;
ltilde_g = zeros(P,Q);
ltrace_g = ltilde_g;
wAm = cell(P,1); % wAm{i} = \sum_j w_ij A_j(oi) m_j
Ag = cell(Q,1);
for j=1:Q
  [Ag{j},Knm,Kmminv,Lmm] = computeKnmKmminv(cf.covfunc_g, par.g{j}.loghyp, x, par.g{j}.z);
  diagKnn = feval(cf.covfunc_g, par.g{j}.loghyp, x, 'diag');
  lkl = 0.5*(logdetChol(Lmm) - logdet(par.g{j}.S) + traceABsym(Kmminv,par.g{j}.S) ...
    + par.g{j}.m'*Kmminv*par.g{j}.m); % kl part
  elbo = elbo - lkl;
  for i=1:P
    indice = par.idx(:,i);
    betaval = par.beta(i);
    w = par.w(i,j); w2 = w*w;
    ltilde_g(i,j) = 0.5*betaval*w2*sum(diagKnn(indice) - diagProd(Ag{j}(indice,:),Knm(indice,:)')); % Ktilde part
    ltrace_g(i,j) = 0.5*betaval*w2*traceABsym(par.g{j}.S,Ag{j}(indice,:)'*Ag{j}(indice,:)); % trace(S_jA'A) part
    if isempty(wAm{i})
      wAm{i} = w*Ag{j}(indice,:)*par.g{j}.m;
    else
      wAm{i} = wAm{i} + w*Ag{j}(indice,:)*par.g{j}.m;
    end
  end
end
elbo = elbo - sum(sum(ltilde_g)) - sum(sum(ltrace_g));
y0 = cell(P,1);
for i=1:P
  indice = par.idx(:,i);
  y0{i} = y(indice,i) - wAm{i};
  betaval = par.beta(i);
  logN = -0.5*sum(indice)*log(2*pi/betaval) - 0.5*betaval*sum(y0{i}.^2);
  elbo = elbo + logN;
end

dbeta = []; dw = []; dloghyp = []; dz = [];

% derivatives of dbeta,dw,dloghyp
if nargout >= 2 && nargout <= 4
  dbeta = zeros(P,1);
  for i=1:P
    Ni = sum(par.idx(:,i));
    betaval = par.beta(i);
    dbeta(i) = - 0.5*sum(y0{i}.^2) + (0.5*Ni - sum(ltilde_g(i,:)) - sum(ltrace_g(i,:)))/betaval;
  end
  if nargout >= 3   % derivatives of weights, hyper
    dw = cell(Q,1); dloghyp = cell(Q,1);
    for j=1:Q
      dloghyp{j} = zeros(size(par.g{j}.loghyp));
      dw{j} = zeros(P,1);
      for i=1:P
        indice = par.idx(:,i);
        xi = x(indice,:);
        w = par.w(i,j);
        betaval = par.beta(i);
        Amj = Ag{j}(indice,:)*par.g{j}.m;
        % y_i - A_i m_i - \sum_{j' != j} w_ij'A_j(oi) m_j
        %    = y_i - A_i m_i - (wAm{i} - w_ij * Amj)
        %    = y0(i) + w_ij*Amj
        ytmp = y0{i} + w*Amj;
        par.g{j}.beta = betaval;
        par.g{j}.w = w;
        [~,dloghyp_i] = ssvi_elbo_g(xi,ytmp,par.g{j},cf.covfunc_g,P);
        dloghyp{j} = dloghyp{j} + dloghyp_i;
        dw{j}(i) = -(2/w)*ltilde_g(i,j) -(2/w)*ltrace_g(i,j) + betaval*ytmp'*Amj - betaval*w*sum(Amj.^2);
      end
    end
  end
end

if nargout == 5   % derivatives of inducing inputs
  dz = cell(Q,1);
  for j=1:Q
    dz{j} = zeros(size(par.g{j}.z));
    for i=1:P
      indice = par.idx(:,i);
      xi = x(indice,:);
      w = par.w(i,j);
      betaval = par.beta(i);
      Amj = Ag{j}(indice,:)*par.g{j}.m;
      ytmp = y0{i} + w*Amj;
      par.g{j}.beta = betaval;
      par.g{j}.w = w;
      [~,~,dz_i] = ssvi_elbo_g(xi,ytmp,par.g{j},cf.covfunc_g,P);
      dz{j} = dz{j} + dz_i;
    end
  end
end
  
% dm,dS (only for checking gradients)
if nargout >= 6
  dm = cell(Q,1); dS = cell(Q,1);
  for j=1:Q
    [~,~,Lambda,~,~] = computeKnmKmminv(cf.covfunc_g, par.g{j}.loghyp, x, par.g{j}.z);
    tmp = zeros(size(par.g{j}.m));
    for i=1:P
      w = par.w(i,j); w2 = w*w;
      betaval = par.beta(i);
      indice = par.idx(:,i);
      Lambda = Lambda + betaval*w2*(Ag{j}(indice,:)')*Ag{j}(indice,:);
      ytmp = y0{i} + w*Ag{j}(indice,:)*par.g{j}.m;
      tmp = tmp + betaval*w*Ag{j}(indice,:)'*ytmp;
    end
    Sinv = invChol(jit_chol(par.g{j}.S));
    dm{j} = tmp - Lambda*par.g{j}.m;
    dS{j} = 0.5*Sinv - 0.5*Lambda;
  end
end

end

