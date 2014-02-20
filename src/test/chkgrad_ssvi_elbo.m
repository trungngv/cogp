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
N = 100; M = 5; D = 2; P = 2;
cf.covfunc_g = 'covSEard';
cf.covfunc_h = 'covSEard';
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
params.idx = ~isnan(y);
% init params for g and h_i
params.g = init_params(x,rand(N,1),M,nhyper_g,0,z_g);
params.task = cell(P,1);
for i=1:P
  z = 10*rand(M,D);
  params.task{i} = init_params(x,rand(N,1),M,nhyper_h,0,z);
end

% test for hyperparameters of g
theta = params.g.loghyp;
[mygrad, delta] = gradchek(theta', @elbo_g_hyp, @delbo_g_hyp, x,y,params,cf);
Kmm = feval(cf.covfunc_g, params.g.loghyp, z);
myassert(mygrad,delta,rcond(Kmm),'test ssvi_elbo() for hyperparamters of g passed');

% test for inducing inputs of g
theta = params.g.z(:);
[mygrad, delta] = gradchek(theta', @elbo_g_z, @delbo_g_z, x,y,params,cf);
myassert(mygrad,delta,rcond(Kmm),'test ssvi_elbo() for inducing inputs of g passed');

% test for m_g
% theta = params.g.m;
% [mygrad,delta] = gradchek(theta', @elbo_g_m, @delbo_g_m, x,y,params,cf);
% myassert(mygrad,delta,rcond(Kmm),'test ssvi_elbo() for m_g passed');

% test for hyperparamters of h_i 
for i=1:P
  theta = [params.task{i}.loghyp; params.task{i}.z(:); params.task{i}.beta];
  [mygrad, delta] = gradchek(theta', @elbo_h, @delbo_h, x,y,params,cf,i);
  Kmm = feval(cf.covfunc_h, params.task{i}.loghyp, params.task{i}.z);
  myassert(mygrad,delta,rcond(Kmm),'test ssvi_elbo() for parameters of h_i passed');
end
end

% L_g as a function of hyperparameters
function fval = elbo_g_hyp(theta,x,y,params,cf)
params.g.loghyp = theta';
fval = ssvi_elbo(x,y,params,cf);
end

%dL_g/dhyp
function g = delbo_g_hyp(theta,x,y,params,cf)
params.g.loghyp = theta';
[~,gloghyp,~] = ssvi_elbo(x,y,params,cf);
g = gloghyp';
end

% L_g as a function of inducing inputs
function fval = elbo_g_z(theta,x,y,params,cf)
z = reshape(theta,numel(params.g.m),[]);
params.g.z = z;
fval = ssvi_elbo(x,y,params,cf);
end

% dL_g/dz_g
function g = delbo_g_z(theta,x,y,params,cf)
z = reshape(theta,numel(params.g.m),[]);
params.g.z = z;
[~,~,gz] = ssvi_elbo(x,y,params,cf);
g = gz(:)';
end

% L_g as a function of m_g
function fval = elbo_g_m(theta,x,y,params,cf)
params.g.m = theta';
fval = ssvi_elbo(x,y,params,cf);
end

% dL_g/dm
function g = delbo_g_m(theta,x,y,params,cf)
params.g.m = theta';
[~,~,~,gm] = ssvi_elbo(x,y,params,cf);
g = gm(:)';
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
Kmm = feval(cf.covfunc_g, params.g.loghyp, params.g.z);
Knm = feval(cf.covfunc_g, params.g.loghyp, x(indice,:), params.g.z);
diagKnn = feval(cf.covfunc_g, params.g.loghyp, x, 'diag');
Lmm = jit_chol(Kmm,4);
Kmminv = invChol(Lmm);
A = Knm*Kmminv;
S_g = params.g.S;
y_discounted = y(indice,i) - A*params.g.m;
[~,dloghyp,dbeta,dz] = svi_elbo(x(indice,:),y_discounted,params.task{i},cf.covfunc_h,[],[]);
dbeta_g = - 0.5*sum(diagKnn(indice)) - 0.5*traceABsym(S_g,A'*A); % contribution from g
dbeta = dbeta + dbeta_g;
g = [dloghyp; dz(:); dbeta]';
end

function myassert(mygrad,delta,rconditional,msg)
  delta = abs(delta);
  [valDiff,idx] = max(delta);
  percentageDiff = (valDiff*100/abs(mygrad(idx)));
  if (valDiff > 1e-5 && percentageDiff > 5)
    fprintf('test ssvi_elbo() failed with valDiff = %.4f\n', valDiff);
    fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
    fprintf('rcond = %.10f\n', rconditional);
  else
    disp(msg);
  end
end

