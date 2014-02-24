function chkgrad_svi_elbo()
%CHKGRADELBO chkgrad_svi_elbo()
%  Check gradient of the evidence lowerbound wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   svi_elbo

%rng(1110, 'twister');
N = 100; M = 10; D = 3;
x = 5*rand(N,D);
y = 2*rand(N,1);
cf.covfunc = 'covSEard';
nhyper = eval(feval(cf.covfunc));
hypPeriodic = log([0.9;2;2]);
hypSEard = log([rand(nhyper-1,1); 2]);
hyp = log(rand(nhyper,1));
params.z = 10*rand(M,D);
params.m = rand(M,1);
L = rand(M,M);
params.S = L'*L;
params.beta = 1e-2;
params.loghyp = hypSEard;
Kmm = feval(cf.covfunc, params.loghyp, params.z);
rconditional = rcond(Kmm);

theta = [params.loghyp; params.beta; params.z(:)];
[mygrad, delta] = gradchek(theta', @f, @grad, x,y,params,cf);
assert_helper(mygrad,delta,rconditional, 'gradient check for svi_elbo() passed');
end

function fval = f(theta,x,y,params,cf)
nhyp = numel(params.loghyp);
params.loghyp = theta(1:nhyp)';
params.beta = theta(nhyp+1);
params.z = reshape(theta(nhyp+2:end),numel(params.m),[]);
fval = svi_elbo(x,y,params,cf.covfunc);
end

function g = grad(theta,x,y,params,cf)
nhyp = numel(params.loghyp);
params.loghyp = theta(1:nhyp)';
params.beta = theta(nhyp+1);
params.z = reshape(theta(nhyp+2:end),numel(params.m),[]);
[~,dloghyp,dbeta,dz] = svi_elbo(x,y,params,cf.covfunc);
g = [dloghyp; dbeta; dz(:)]';
end

function assert_helper(mygrad,delta,rconditional,msg)
  delta = abs(delta);
  [valDiff,idx] = max(delta);
  percentageDiff = (valDiff*100/abs(mygrad(idx)));
  if (valDiff > 1e-4)
    fprintf('test ssvi_elbo() failed with valDiff = %.5f\n', valDiff);
    fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
    fprintf('rcond = %.10f\n', rconditional);
  else
    disp(msg);
  end
end

