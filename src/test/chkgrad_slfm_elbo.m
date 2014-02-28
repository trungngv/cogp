function chkgrad_slfm_elbo()
%CHKGRAD_SLFM_ELBO chkgrad_slfm_elbo()
%  Check gradient of the ELBO wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   slfm_elbo

%rng(1110, 'twister');

% note on gradient:
% 1. delbo/dgj is from slfm_elbo
% 2. delbo/dhi is from svi_elbo with y discounted by contribution from g 
%
N = 20; M = 5; D = 3; P = 3; Q = 3;
cf.covfunc_g = 'covSEard';
cf.covfunc_h = 'covSEard';
nhyper_g = eval(feval(cf.covfunc_g));
nhyper_h = eval(feval(cf.covfunc_h));
cf.n_outputs = P;
x = 10*rand(N,D);
y1 = 2*rand(N,1);
y2 = 2*rand(N,1);
y3 = 2*rand(N,1);
Nmiss = 3;
randind = randperm(N,3*Nmiss);
y1(randind(1:Nmiss)) = nan;
y2(randind(Nmiss+1:Nmiss*2)) = nan;
y3(randind(Nmiss*2+1:end)) = nan;
y = [y1,y2,y3];

% init params for g_j and h_i
par.task = cell(P,1);
par.g = cell(Q,1);
par.beta = rand(P,1);
for i=1:P
  z = 10*rand(M,D);
  par.task{i} = init_params(x,rand(N,1),M,nhyper_h,0,z);
end
for j=1:Q
  z = 10*rand(M,D);
  par.g{j} = init_params(x,rand(N,1),M,nhyper_g,0,z);  
end
par.idx = ~isnan(y);
par.w = 0.1+rand(P,Q);

% test for hyperparameters and inducing inputs of all g_j
theta = [];
for j=1:Q
  theta = [theta; par.g{j}.loghyp; par.w(:,j); par.g{j}.z(:)];
end
for i=1:P
  theta = [theta; par.beta(i)];
end
[mygrad, delta] = gradchek(theta', @elbo_g, @delbo_g, x,y,par,cf);
assert_helper(mygrad,delta,'test slfm_elbo() for parameters of g passed');
disp('press any key...')
pause

% test for m_g,S_g
theta = [];
for j=1:Q
  theta = [theta; par.g{j}.m; par.g{j}.S(:)];
end
[mygrad,delta] = gradchek(theta', @elbo_g2, @delbo_g2, x,y,par,cf);
assert_helper(mygrad,delta,'test slfm_elbo() for m_j,S_j passed');
disp('press any key...')
pause

% test for hyperparamters of h_i 
theta = [];
for i=1:P
  theta = [par.task{i}.loghyp; par.task{i}.z(:)];
  [mygrad, delta] = gradchek(theta', @elbo_h, @delbo_h, x,y,par,cf,i);
  assert_helper(mygrad,delta,'test slfm_elbo() for parameters of h_i passed');
end
end

function fval = elbo_g(theta,x,y,par,cf)
P = numel(par.task); Q = numel(par.g);
noffset = 0; 
for j=1:Q
  nhyp = numel(par.g{j}.loghyp);
  par.g{j}.loghyp = theta(noffset+1:noffset + nhyp)';
  par.w(:,j) = theta(noffset+nhyp+1:noffset+nhyp+P)';
  par.g{j}.z = reshape(theta(noffset+nhyp+P+1:noffset+nhyp+P+numel(par.g{j}.z)),numel(par.g{j}.m),[]);
  noffset = noffset + nhyp + P + numel(par.g{j}.z);
end
for i=1:P
  par.beta(i) = theta(noffset+i);
end
fval = slfm_elbo(x,y,par,cf);
end

function g = delbo_g(theta,x,y,par,cf)
P = numel(par.task); Q = numel(par.g);
noffset = 0; 
for j=1:Q
  nhyp = numel(par.g{j}.loghyp);
  par.g{j}.loghyp = theta(noffset+1:noffset+nhyp)';
  par.w(:,j) = theta(noffset+nhyp+1:noffset+nhyp+P)';
  par.g{j}.z = reshape(theta(noffset+nhyp+P+1:noffset+nhyp+P+numel(par.g{j}.z)),numel(par.g{j}.m),[]);
  noffset = noffset + nhyp + P + numel(par.g{j}.z);
end
for i=1:P
  par.beta(i) = theta(noffset+i);
end
[~,dbeta,dw,dloghyp,dz] = slfm_elbo(x,y,par,cf);
g = [];
for j=1:Q
  g = [g; dloghyp{j}; dw{j}; dz{j}(:)];
end
g = [g; dbeta];
g = g';
end

% L_g as a function of m_g,S_g
function fval = elbo_g2(theta,x,y,par,cf)
Q = numel(par.g);
noffset = 0;
for j=1:Q
  M = numel(par.g{j}.m);
  par.g{j}.m = theta(noffset+1:noffset+M)';
  par.g{j}.S = reshape(theta(noffset+M+1:noffset+M+M*M),M,M);
  noffset = noffset+M+M*M;
end
fval = slfm_elbo(x,y,par,cf);
end

% dL_g/dm,dL_g/dS
function g = delbo_g2(theta,x,y,par,cf)
Q = numel(par.g);
noffset = 0;
for j=1:Q
  M = numel(par.g{j}.m);
  par.g{j}.m = theta(noffset+1:noffset+M)';
  par.g{j}.S = reshape(theta(noffset+M+1:noffset+M+M*M),M,M);
  noffset = noffset+M+M*M;
end
[fval,~,~,~,~,dm,dS] = slfm_elbo(x,y,par,cf);
g = [];
for j=1:Q
  g = [g; dm{j}; dS{j}(:)];
end
g = g';
end

% L_i as a function of paramters of h_i
function fval = elbo_h(theta,x,y,par,cf,i)
nhyp = numel(par.task{i}.loghyp);
D = size(x,2);
par.task{i}.loghyp = theta(1:nhyp)';
par.task{i}.z = reshape(theta(nhyp+1:end),[],D);
fval = slfm_elbo(x,y,par,cf);
end

function g = delbo_h(theta,x,y,par,cf,i)
nhyp = numel(par.task{i}.loghyp);
par.task{i}.loghyp = theta(1:nhyp)';
D = size(x,2);
par.task{i}.z = reshape(theta(nhyp+1:end),[],D);
indice = par.idx(:,i);
wAm = zeros(sum(indice),1); % wAm = \sum_j w_ij A_j(oi) m_j
for j=1:numel(par.g)
  A = computeKnmKmminv(cf.covfunc_g, par.g{j}.loghyp, x(indice,:), par.g{j}.z);
  w = par.w(i,j);
  wAm = wAm + w*A*par.g{j}.m;
end  
y0 = y(indice,i) - wAm;
par.task{i}.beta = par.beta(i);
[~,dloghyp,~,dz] = svi_elbo(x(indice,:),y0,par.task{i},cf.covfunc_h);
g = [dloghyp; dz(:)]';
end

function assert_helper(mygrad,delta,msg)
  delta = abs(delta);
  [valDiff,idx] = max(delta);
  percentageDiff = (valDiff*100/abs(mygrad(idx)));
  if (valDiff > 1e-4)
    fprintf('test ssvi_elbo() failed with valDiff = %.5f\n', valDiff);
    fprintf('percentage (valDiff / gradient) = %.2f\n', percentageDiff);
  else
    disp(msg);
  end
end

