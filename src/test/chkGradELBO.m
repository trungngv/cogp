function chkGradELBO()
%CHKGRADELBO chkGradELBO()
%  Check gradient of the evidence lowerbound wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   sviELBO

%rng(1110, 'twister');
N = 100; M = 20; D = 1;
x = 10*rand(N,D);
y = 2*rand(N,1);
z = 10*rand(M,D);
m = rand(M,1);
L = rand(M,M);
S = L'*L;
cf.betaval = 1e-5;
cf.covfunc = 'covSEard';
cf.loghyp = [log(rand(D,1)); 1+rand];
theta = [cf.loghyp; cf.betaval];
[mygrad, delta] = gradchek(theta', @f, @grad, x,y,z,m,S,cf);
delta = abs(delta);
[valDiff,idx] = max(delta);
percentageDiff = (valDiff*100/abs(mygrad(idx)));
if (valDiff > 1e-5 && percentageDiff > 10)
  fprintf('test sviELBO() failed with valDiff = %.4f\n', valDiff);
  fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
  Kmm = feval(cf.covfunc, cf.loghyp, z);  
  fprintf('conditional number = %.4f\n', cond(Kmm));
else
  disp('test sviELBO() passed');
end

end

function fval = f(theta,x,y,z,m,S,cf)
  cf.loghyp = theta(1:end-1)';
  cf.betaval = theta(end);
  fval = sviELBO(x,y,z,m,S,cf,[],[],[],[]);
end

function g = grad(theta,x,y,z,m,S,cf)
  cf.loghyp = theta(1:end-1)';
  cf.betaval = theta(end);
  [~, gloghyp,gbeta] = sviELBO(x,y,z,m,S,cf,[],[],[],[]);
  g = [gloghyp; gbeta]';
end

