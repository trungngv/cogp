function [elbo,dloghyp,dz] = ssvi_elbo_g(x,y,params,covfunc,n_outputs)
%SVI_ELBO [elbo,dloghyp,dz] = ssvi_elbo_g(x,y,params,covfunc,n_outputs)
%   
% For g, use only 1/P KL, all other thing same.
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - params : parameters structure
%   - covfunc : covariance function
%   - n_outputs : #outputs (1/P term)
%
% OUTPUT
%   - elbo, dloghyp, dz : objective value and its derivatives
%
% Trung Nguyen
% 17/02/14
z = params.z;
m = params.m;
S = params.S;
betaval = params.beta;
loghyp = params.loghyp;
N = size(x,1);
M = size(z,1);
Kmm = feval(covfunc, loghyp, z);
Lmm = jit_chol(Kmm,3);
Kmminv = invChol(Lmm);
Knm = feval(covfunc, loghyp, x, z);
A = Knm*Kmminv;

diagKnn = feval(covfunc, loghyp, x, 'diag');
yMinusAm = y - A*m;
logN = -0.5*N*log(2*pi/betaval) - 0.5*betaval*sum(yMinusAm.^2); % \sum logN() part
ltilde = 0.5*betaval*sum(diagKnn - diagProd(A,Knm')); % Ktilde part
ltrace = 0.5*betaval*traceABsym(S,A'*A); % trace(SA'A) part
lkl = 0.5*(logdetChol(Lmm) - logdet(S) + traceABsym(Kmminv,S) + m'*Kmminv*m - M); % kl-divergence part
elbo = logN - ltilde - ltrace - lkl/n_outputs;

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
    dloghyp(i) = dlogN - dTilde - dTrace - dKL/n_outputs;
  end
end

if nargout == 3   % inducing inputs derivatives
  dz = zeros(size(z));
  ell2 = exp(2*loghyp(1:end-1));
  ASKmminv = A*S*Kmminv;
  D1 = yMinusAm*m'*Kmminv + A - ASKmminv;
  D1 = betaval*D1;
  D2 = -betaval*Kmminv*m*yMinusAm'*A - 0.5*betaval*(A')*A + betaval*ASKmminv'*A;
  D2 = D2 - 0.5*Kmminv/n_outputs + 0.5*Kmminv*(m*m'+S)*Kmminv/n_outputs;
  for i=1:size(z,2)
    DKmm = bsxfun(@minus,z(:,i),z(:,i)');
    DKmm = -(Kmm.*DKmm)/ell2(i);
    DKmn = bsxfun(@minus,z(:,i),x(:,i)');
    DKmn = -(Knm'.*DKmn)/ell2(i);
    dz(:,i) = sum(D1'.*DKmn,2) + sum(D2.*DKmm,2) + sum(D2'.*DKmm,2) - diag(D2).*diag(DKmm);
  end
end

end
