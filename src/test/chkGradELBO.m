function chkGradELBO()
%CHKGRADELBO chkGradELBO()
%  Check gradient of the evidence lowerbound wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   sviELBO

%rng(1110, 'twister');
N = 1000; M = 50; D = 2;
x = 10*rand(N,D);
y = 2*rand(N,1);
cf.covfunc = 'covSEard';
params.z = 10*rand(M,D);
params.m = rand(M,1);
L = rand(M,M);
params.S = L'*L;
params.beta = 1e-2;
params.loghyp = [log(rand(D,1)); 1+rand];
% test for hyperparameters
theta = [params.loghyp; params.beta];
[mygrad, delta] = gradchek(theta', @f, @grad, x,y,params,cf);
delta = abs(delta);
[valDiff,idx] = max(delta);
percentageDiff = (valDiff*100/abs(mygrad(idx)));
if (valDiff > 1e-5 && percentageDiff > 10)
  fprintf('test sviELBO() failed with valDiff = %.4f\n', valDiff);
  fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
  Kmm = feval(cf.covfunc, cf.loghyp, z);  
  fprintf('rcond = %.10f\n', rcond(Kmm));
else
  disp('test sviELBO() passed');
end

% test for inducing inputs
theta = params.z(:);
[mygrad, delta] = gradchek(theta', @fz, @gradz, x,y,params,cf);
delta = abs(delta);
[valDiff,idx] = max(delta);
percentageDiff = (valDiff*100/abs(mygrad(idx)));
if (valDiff > 1e-4 && percentageDiff > 10)
  fprintf('test sviELBO() failed with valDiff = %.4f\n', valDiff);
  fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
  Kmm = feval(cf.covfunc, cf.loghyp, z);  
  fprintf('rcond = %.10f\n', rcond(Kmm));
else
  disp('test sviELBO() passed');
end

end

function fval = f(theta,x,y,params,cf)
  params.loghyp = theta(1:end-1)';
  params.beta = theta(end);
  fval = sviELBO(x,y,params,cf,[],[],[],[]);
end

function g = grad(theta,x,y,params,cf)
  params.loghyp = theta(1:end-1)';
  params.beta = theta(end);
  [~, gloghyp,gbeta] = sviELBO(x,y,params,cf,[],[],[],[]);
  g = [gloghyp; gbeta]';
end

function fval = fz(theta,x,y,params,cf)
  z = reshape(theta,numel(params.m),[]);
  params.z = z;
  fval = sviELBO(x,y,params,cf,[],[],[],[]);
end

function g = gradz(theta,x,y,params,cf)
  z = reshape(theta,numel(params.m),[]);
  params.z = z;
  [~,~,~,gz] = sviELBO(x,y,params,cf,[],[],[],[]);
  g = gz(:)';
end
