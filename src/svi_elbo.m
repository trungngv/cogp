function [elbo,dloghyp,dbeta,dz] = svi_elbo(x,y,params,covfunc,Lmm,Kmminv,Knm,A)
%SVI_ELBO [elbo,dloghyp,dbeta,dz] = svi_elbo(x,y,params,covfunc,Lmm,Kmminv,Knm,A)
%   
% Lowerbound as a function of the hyperparameters (and inducing inputs) and
% their derivatives. Note that the inducing derivatives are for the
% SEard covariance function only.
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - params : parameters structure
%   - covfunc : covariance function
%   - Lmm, Kmminv, Knm, A : [optional] previously computed values of those
%   (see also SVIUPDATE).
%
% OUTPUT
%   - elbo, dloghyp, dbeta, dz : objective value and its derivatives
%
% Trung Nguyen
% 18/02/14
z = params.z;
m = params.m;
S = params.S;
betaval = params.beta;
loghyp = params.loghyp;
N = size(x,1);
M = size(z,1);
Kmm = feval(covfunc, loghyp, z);
if isempty(Lmm)
  Lmm = jit_chol(Kmm,3);
  Kmminv = invChol(Lmm);
  Knm = feval(covfunc, loghyp, x, z);
  A = Knm*Kmminv;
end

%TODO optimize code if possible
diagKnn = feval(covfunc, loghyp, x, 'diag');
yMinusAm = y - A*m;
logN = -0.5*N*log(2*pi/betaval) - 0.5*betaval*sum(yMinusAm.^2); % \sum logN() part
ltilde = 0.5*betaval*sum(diagKnn - diagProd(A,Knm')); % Ktilde part
ltrace = 0.5*betaval*traceABsym(S,A'*A); % trace(SA'A) part
lkl = 0.5*(logdetChol(Lmm) - logdet(S) + traceABsym(Kmminv,S) + m'*Kmminv*m - M); % kl-divergence part
elbo = logN - ltilde - ltrace - lkl;
%logLTrue = l3bound(y,Knm,Lmm,Kmminv,diagKnn - diagProd(A,Knm'),betaval,m,S);

if nargout >= 2     %covariance derivatives
  dloghyp = zeros(size(params.loghyp));
  for i=1:numel(params.loghyp)
    % covSEard returns dK / dloghyper 
    dKnm = feval(covfunc, loghyp, x, z, i);
    dKmm = feval(covfunc, loghyp, z, [], i);
    dKnn = feval(covfunc, loghyp, x, 'diag', i);
    dA = (dKnm - A*dKmm)*Kmminv;
    dlogN = betaval*(yMinusAm'*dA)*m;
    dTilde = 0.5*betaval*(sum(dKnn) - sum(diagProd(A,dKnm')) - sum(diagProd(dA,Knm')));
    dTrace = betaval*traceAB(A*S,dA');
    dKL = 0.5*(traceABsym(Kmminv,dKmm) - traceABsym(Kmminv,dKmm*Kmminv*(m*m'+S)));
    dloghyp(i) = dlogN - dTilde - dTrace - dKL;
  end
end

if nargout >= 3   % noise derivative
  dbeta = 0.5*N/betaval - 0.5*sum(yMinusAm.^2) -ltilde/betaval - ltrace/betaval;
end

if nargout == 4   % inducing inputs derivatives
  dz = zeros(size(z));
  ell2 = exp(2*loghyp(1:end-1));
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