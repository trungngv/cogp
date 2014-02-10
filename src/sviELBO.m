function [logL,dloghyp] = sviELBO(x,y,z,m,S,cf,Lmm,Kmminv,Knm,A)
%SVIUPDATE [fval,dloghyp] = sviELBO(x,y,z,m,S,mem,cf)
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
betaval = cf.betaval;
if isempty(Lmm)
  Kmm = feval(cf.covfunc, cf.loghyp, z);
  fprintf('conditional number = %.4f\n', cond(Kmm));
  Lmm = jit_chol(Kmm,3); clear Kmm;
  Kmminv = invChol(Lmm);
  Knm = feval(cf.covfunc, cf.loghyp, x, z);
  A = Knm*Kmminv;
end

%TODO
% 1) optimize some code if possible
% 2) remove terms independent of hypers: S, M, N,pi
diagKnn = feval(cf.covfunc, cf.loghyp, x, 'diag');
logL = N*log(2*pi/betaval) + betaval*sum((y - A*m).^2); % \sum logN() part
logL = logL + betaval*sum(diagKnn - diagProd(A,Knm')); % Ktilde part
logL = logL + betaval*traceABsym(S,A'*A); % trace(SA'A) part
logL = logL + logdetChol(Lmm) - logdet(S) + traceABsym(Kmminv,S) + m'*Kmminv*m - M; % kl-divergence part
logL = -0.5*logL;
%logLTrue = l3bound(y,Knm,Lmm,Kmminv,diagKnn - diagProd(A,Knm'),betaval,m,S);

if nargout == 2
  dloghyp = zeros(size(cf.loghyp));
  for i=1:numel(cf.loghyp)
    % covSEard returns dK / dloghyper 
    dKnm = feval(cf.covfunc, cf.loghyp, x, z, i);
    dKmm = feval(cf.covfunc, cf.loghyp, z, [], i);
    dKnn = feval(cf.covfunc, cf.loghyp, x, 'diag', i);
    dA = (dKnm - A*dKmm)*Kmminv;
    dlogN = -betaval*((A*m-y)'*dA)*m;
    dTilde = 0.5*betaval*(sum(dKnn) - sum(diagProd(A,dKnm')) - sum(diagProd(dA,Knm')));
    dTrace = betaval*traceAB(A*S,dA');
    dKL = 0.5*(traceABsym(Kmminv,dKmm) - traceABsym(Kmminv,dKmm*Kmminv*(m*m'+S)));
    dloghyp(i) = dlogN - dTilde - dTrace - dKL;
  end
end

return;

