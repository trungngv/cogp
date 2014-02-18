function chkgrad_svi_elbo()
%CHKGRADELBO chkgrad_svi_elbo()
%  Check gradient of the evidence lowerbound wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   svi_elbo

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
  fprintf('test svi_elbo() failed with valDiff = %.4f\n', valDiff);
  fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
  Kmm = feval(cf.covfunc, params.loghyp, z);  
  fprintf('rcond = %.10f\n', rcond(Kmm));
else
  disp('test svi_elbo() passed');
end

% test for inducing inputs
theta = params.z(:);
[mygrad, delta] = gradchek(theta', @fz, @gradz, x,y,params,cf);
delta = abs(delta);
[valDiff,idx] = max(delta);
percentageDiff = (valDiff*100/abs(mygrad(idx)));
if (valDiff > 1e-4 && percentageDiff > 10)
  fprintf('test svi_elbo() failed with valDiff = %.4f\n', valDiff);
  fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
  Kmm = feval(cf.covfunc, cf.loghyp, z);  
  fprintf('rcond = %.10f\n', rcond(Kmm));
else
  disp('test svi_elbo() passed');
end

end

function fval = f(theta,x,y,params,cf)
  params.loghyp = theta(1:end-1)';
  params.beta = theta(end);
  fval = svi_elbo(x,y,params,cf.covfunc,[],[],[],[]);
end

function g = grad(theta,x,y,params,cf)
  params.loghyp = theta(1:end-1)';
  params.beta = theta(end);
  [~, gloghyp,gbeta] = svi_elbo(x,y,params,cf.covfunc,[],[],[],[]);
  g = [gloghyp; gbeta]';
end

function fval = fz(theta,x,y,params,cf)
  z = reshape(theta,numel(params.m),[]);
  params.z = z;
  fval = svi_elbo(x,y,params,cf.covfunc,[],[],[],[]);
end

function g = gradz(theta,x,y,params,cf)
  z = reshape(theta,numel(params.m),[]);
  params.z = z;
  [~,~,~,gz] = svi_elbo(x,y,params,cf.covfunc,[],[],[],[]);
  g = gz(:)';
end
