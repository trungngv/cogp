function chkgrad_ssvi_elbo()
%CHKGRAD_SSVI_ELBO chkgrad_ssvi_elbo()
%  Check gradient of the evidence lowerbound wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   ssvi_elbo

%rng(1110, 'twister');

% note on gradient:
% 1. delbo/dg is from ssvi_elbo
% 2. delbo/dhi is from svi_elbo except that y is discounted contribution
% from g and beta also has contribution from g.
% TODO: output derivatives wrt h_i directly from ssvi_elbo.
%
N = 100; M = 30; D = 2; P = 2;
cf.covfunc_g = 'covSEard';
cf.covfunc_h = 'covNoise';
cf.use_h = false;
nhyper_g = eval(feval(cf.covfunc_g));
nhyper_h = eval(feval(cf.covfunc_h));
cf.n_outputs = P;
x = 10*rand(N,D);
y1 = 2*rand(N,1);
y2 = 2*rand(N,1);
Nmiss = 10;
randind = randperm(N,2*Nmiss);
y1(randind(1:Nmiss)) = nan;
y2(randind(Nmiss+1:end)) = nan;
y = [y1,y2];
z_g = 10*rand(M,D);
% init params for g and h_i
params.g = init_params(x,rand(N,1),M,nhyper_g,0,z_g);
params.task = cell(P,1);
for i=1:P
  z = 10*rand(M,D);
  params.task{i} = init_params(x,rand(N,1),M,nhyper_h,0,z);
end
params.idx = ~isnan(y);
params.w = 0.1+rand(P,1);
Kmm = feval(cf.covfunc_g, params.g.loghyp, z);

% test for inducing inputs of g
theta = [params.g.loghyp; params.w; params.g.z(:)];
[mygrad, delta] = gradchek(theta', @elbo_g, @delbo_g, x,y,params,cf);
assert_helper(mygrad,delta,rcond(Kmm),'test ssvi_elbo() for parameters of g passed');

% test for m_g,S_g
% theta = [params.g.m; params.g.S(:)];
% [mygrad,delta] = gradchek(theta', @elbo_g, @delbo_g, x,y,params,cf);
% assert_helper(mygrad,delta,rcond(Kmm),'test ssvi_elbo() for m_g passed');

% test for hyperparamters of h_i 
for i=1:P
  theta = [params.task{i}.loghyp; params.task{i}.z(:); params.task{i}.beta];
  [mygrad, delta] = gradchek(theta', @elbo_h, @delbo_h, x,y,params,cf,i);
  Kmm = feval(cf.covfunc_h, params.task{i}.loghyp, params.task{i}.z);
  assert_helper(mygrad,delta,rcond(Kmm),'test ssvi_elbo() for parameters of h_i passed');
end
end

function fval = elbo_g(theta,x,y,params,cf)
nhyp = numel(params.g.loghyp);
P = numel(params.task);
params.g.loghyp = theta(1:nhyp)';
params.w = theta(nhyp+1:nhyp+P)';
params.g.z = reshape(theta(nhyp+P+1:end),numel(params.g.m),[]);
fval = ssvi_elbo(x,y,params,cf);
end

function g = delbo_g(theta,x,y,params,cf)
nhyp = numel(params.g.loghyp);
P = numel(params.task);
params.g.loghyp = theta(1:nhyp)';
params.w = theta(nhyp+1:nhyp+P)';
params.g.z = reshape(theta(nhyp+P+1:end),numel(params.g.m),[]);
[~,dloghyp,dw,dz] = ssvi_elbo(x,y,params,cf);
g = [dloghyp; dw; dz(:)]';
end

% L_g as a function of m_g,S_g
function fval = elbo_g2(theta,x,y,params,cf)
M = numel(params.g.m);
params.g.m = theta(1:M)';
params.g.S = reshape(theta(M+1:end),M,M);
fval = ssvi_elbo(x,y,params,cf);
end

% dL_g/dm,dL_g/dS
function g = delbo_g2(theta,x,y,params,cf)
M = numel(params.g.m);
params.g.m = theta(1:M)';
params.g.S = reshape(theta(M+1:end),M,M);
[~,~,~,dm,dS] = ssvi_elbo(x,y,params,cf);
g = [dm; dS(:)]';
end

% L_i as a function of paramters of h_i
function fval = elbo_h(theta,x,y,params,cf,i)
nhyp = numel(params.task{i}.loghyp);
D = size(x,2);
params.task{i}.loghyp = theta(1:nhyp)';
params.task{i}.z = reshape(theta(nhyp+1:end-1),[],D);
params.task{i}.beta = theta(end);
fval = ssvi_elbo(x,y,params,cf);
end

function g = delbo_h(theta,x,y,params,cf,i)
nhyp = numel(params.task{i}.loghyp);
params.task{i}.loghyp = theta(1:nhyp)';
D = size(x,2);
params.task{i}.z = reshape(theta(nhyp+1:end-1),[],D);
params.task{i}.beta = theta(end);
indice = params.idx(:,i);
diagKnn = feval(cf.covfunc_g, params.g.loghyp, x, 'diag');
[A,Knm] = computeKnmKmminv(cf.covfunc_g, params.g.loghyp, x(indice,:), params.g.z);
S_g = params.g.S;
w = params.w(i); w2 = w*w;
y_minus_g = y(indice,i) - w*A*params.g.m;
[~,dloghyp,dbeta,dz] = svi_elbo(x(indice,:),y_minus_g,params.task{i},cf.covfunc_h);
dbeta_g = -0.5*w2*sum(diagKnn(indice)-diagProd(A,Knm'));
dbeta_g = dbeta_g - 0.5*w2*traceABsym(S_g,(A')*A);
dbeta = dbeta + dbeta_g;
g = [dloghyp; dz(:); dbeta]';
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

