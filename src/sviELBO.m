function [logL,dloghyp,dbeta,dz] = sviELBO(x,y,z,m,S,cf,Lmm,Kmminv,Knm,A)
%SVIELBO [fval,dloghyp,dbeta,dz] = sviELBO(x,y,z,m,S,cf,Lmm,Kmminv,Knm,A)
%   
% Lowerbound as a function of the hyperparameters (and inducing inputs) and
% their derivatives.
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - z : inducing inputs
%   - m,S : current variational parameters
%   - cf : options for the optimization procedure and hyperparameters
%   - Lmm, Kmminv, Knm, A : [optional] previously computed values of those
%   (see also SVIUPDATE).
%
% OUTPUT
%   - logL, dloghyp : objective value and its derivatives
%
% Trung Nguyen
% 10/02/14
N = size(x,1);
M = size(z,1);
betaval = cf.beta;
if isempty(Lmm)
  Kmm = feval(cf.covfunc, cf.loghyp, z);
  Lmm = jit_chol(Kmm,3);
  Kmminv = invChol(Lmm);
  Knm = feval(cf.covfunc, cf.loghyp, x, z);
  A = Knm*Kmminv;
end

%TODO
% 1) optimize code if possible
% 2) remove terms independent of hypers: S, M, N,pi
diagKnn = feval(cf.covfunc, cf.loghyp, x, 'diag');
yMinusAm = y - A*m;
logN = -0.5*N*log(2*pi/betaval) - 0.5*betaval*sum(yMinusAm.^2); % \sum logN() part
logTilde = 0.5*betaval*sum(diagKnn - diagProd(A,Knm')); % Ktilde part
logTrace = 0.5*betaval*traceABsym(S,A'*A); % trace(SA'A) part
logKL = 0.5*(logdetChol(Lmm) - logdet(S) + traceABsym(Kmminv,S) + m'*Kmminv*m - M); % kl-divergence part
logL = logN - logTilde - logTrace - logKL;
%logLTrue = l3bound(y,Knm,Lmm,Kmminv,diagKnn - diagProd(A,Knm'),betaval,m,S);

if nargout >= 2
  dloghyp = zeros(size(cf.loghyp));
  for i=1:numel(cf.loghyp)
    % covSEard returns dK / dloghyper 
    dKnm = feval(cf.covfunc, cf.loghyp, x, z, i);
    dKmm = feval(cf.covfunc, cf.loghyp, z, [], i);
    dKnn = feval(cf.covfunc, cf.loghyp, x, 'diag', i);
    dA = (dKnm - A*dKmm)*Kmminv;
    dlogN = betaval*(yMinusAm'*dA)*m;
    dTilde = 0.5*betaval*(sum(dKnn) - sum(diagProd(A,dKnm')) - sum(diagProd(dA,Knm')));
    dTrace = betaval*traceAB(A*S,dA');
    dKL = 0.5*(traceABsym(Kmminv,dKmm) - traceABsym(Kmminv,dKmm*Kmminv*(m*m'+S)));
    dloghyp(i) = dlogN - dTilde - dTrace - dKL;
  end
  % noise precision betaval
  dbeta = 0.5*N/betaval - 0.5*sum(yMinusAm.^2) -logTilde/betaval - logTrace/betaval;
  
  % inducing inputs
  dz = zeros(size(z));
  ell2 = exp(2*cf.loghyp(1:end-1));
  ASKmminv = A*S*Kmminv;
  D1 = yMinusAm*m'*Kmminv + A - ASKmminv;
  D1 = betaval*D1;
  D2 = -betaval*Kmminv*m*yMinusAm'*A - 0.5*betaval*(A')*A + betaval*ASKmminv'*A;
  D2 = D2 - 0.5*Kmminv + 0.5*Kmminv*(m*m'+S)*Kmminv;
  for i=1:size(z,2)
    DKmm = bsxfun(@minus,z(:,i),z(:,i)');
    DKmm = -(Kmm.*DKmm)/ell2(i);
    DKmn = bsxfun(@minus,z(:,i),x(:,i)');
    DKmn = -(Knm'.*DKmn)/ell2(i);
    dz(:,i) = sum(D1'.*DKmn,2) + sum(D2.*DKmm,2) + sum(D2'.*DKmm,2) - diag(D2).*diag(DKmm);
  end
end

return;
end

% non-vectorized version of inducing derivatives for sanity check
function dz = dzslow(D1,D2,Kmm,Knm,M,N,x,z,ell2)
  dz = zeros(size(z));
  for m=1:size(z,1)
    for j=1:size(z,2)
      DKmm = zeros(M,M);
      DKmm(:,m) = repmat(z(m,j),M,1) - z(:,j);
      DKmm(m,:) = DKmm(:,m)';
      DKmm = -(Kmm.*DKmm)/ell2(j);
      DKmn = zeros(M,N);
      DKmn(m,:) = repmat(z(m,j),1,N) - x(:,j)';
      DKmn = -(Knm'.*DKmn)/ell2(j);
      dz(m,j) = trace(D1*DKmn) + trace(D2*DKmm);
    end
  end
end
