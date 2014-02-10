function chkGradELBO()
%CHKGRADELBO chkGradELBO()
%  Check gradient of the evidence lowerbound wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   sviELBO

%rng(1110, 'twister');
N = 50; M = 10; D = 1;
x = 5*rand(N,D);
y = rand(N,1);
z = 5*rand(M,D);
m = rand(M,1);
L = rand(M,M);
S = L'*L;
cf.betaval = 1e-5;
cf.covfunc = 'covSEard';
cf.loghyp = log(rand(D+1,1));
[~, delta] = gradchek(cf.loghyp', @f, @grad, x,y,z,m,S,cf);
maxDiff = max(abs(delta));
assert(maxDiff < 1e-5,...
    ['test sviELBO() failed with max diff ' num2str(maxDiff)]);
disp('test sviELBO() passed');

end

function fval = f(loghyp,x,y,z,m,S,cf)
  cf.loghyp = loghyp;
  fval = sviELBO(x,y,z,m,S,cf,[],[],[],[]);
end

function g = grad(loghyp,x,y,z,m,S,cf)
  cf.loghyp = loghyp;
  [~, g] = sviELBO(x,y,z,m,S,cf,[],[],[],[]);
end

