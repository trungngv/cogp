function [mu vaar] = gpsvi_predict(covfunc, loghyper, m, S, z, xstar)
% Makes predictions with gpsvi
% z are the inducing point lcoations
% xstar are the test-pints

Kmm = feval(covfunc, loghyper, z);
Lmm = jit_chol(Kmm,3);
Kmminv = invChol(Lmm);

if size(xstar,1) > 2000
  nbatch = 10;
else
  nbatch = 1;
end
bsize = ceil(size(xstar,1)/nbatch);
mu = [];
vaar= [];
for i=1:nbatch
  pos = (i-1)*bsize + 1;
  if i < nbatch
    bxstar = xstar(pos:i*bsize,:);
  else
    bxstar = xstar(pos:end,:);
  end
  Kss = feval(covfunc, loghyper, bxstar, 'diag');
  Kms = feval(covfunc, loghyper, z, bxstar);
  Ksm = Kms';
  mu = [mu; Ksm*(Kmminv*m)];

  % we can also compute full covariance at a higher cost
  % diag(Ksm * kmminv * S * Kmmonv *Kms) 
  var_1 =  sum(Kms.*(Kmminv*S*Kmminv*Kms),1)';
  var_2 =  sum(Kms.*(Kmminv*Kms),1)';   
  vaar = [vaar; var_1 + Kss - var_2];
end
vaar = max(vaarr,1e-10); % remove numerical noise i.e. negative variance
return;

