function [elbo,par] = slfm_learn(x,y,M,par,cf)
%SLFM_LEARN slfm_learn(x,y,M,cf)
%   Quick wrapper to run structured GPSVI.
% 
% INPUT
%   - x,y : train inputs, train outputs
%   - M : #inducing inputs
%   - cf : configuration
%
% OUTPUT
%   - elbo : the bound
%   - params : learned parameters
[N D] = size(x);
% get observation indice for each output
P = numel(par.task); Q = numel(par.g);
par.w = ones(P,Q);
par.delta_w = zeros(P,Q);
par.beta = (1/0.05)*ones(P,1);
par.delta_beta = zeros(P,1);
observed = ~isnan(y);
nhyper_g = eval(feval(cf.covfunc_g));
for j=1:Q
  par.g{j} = init_params(x,mean(y,2),M.g,nhyper_g,0,[]);
  %params.g.m = zeros(size(params.g.m));
  par.g{j}.beta = []; par.g{j}.delta_beta = [];
end
nhyper_h = eval(feval(cf.covfunc_h));
for i=1:P
  par.task{i} = init_params(x,y(:,i),M.h,nhyper_h,0,[]);
  %params.task{i}.m = zeros(size(params.task{i}.m));
  if strcmp(cf.covfunc_h,'covNoise')
    par.task{i}.loghyp = log(0.01); % learning of noise is hard so set it small here
  end
%  params.task{i}.S = diag(1*ones(numel(params.task{i}.m),1));
end
elbo = zeros(numel(cf.maxiter),1);
for iter = 1:cf.maxiter
  idx = randperm(N, cf.nbatch);
  par.idx = ~isnan(y(idx,:));
  xbatch = x(idx,:); ybatch = y(idx,:);
  
  % variational parameters of g_j
  par = slfm_update_g(xbatch,ybatch,par,cf);
  
  % hyperparameters of g_j
  if cf.learn_z
    [~,dbeta,dw,dloghyp,dz] = slfm_elbo(xbatch,ybatch,par,cf);
  else
    [~,dbeta,dw,dloghyp] = slfm_elbo(xbatch,ybatch,par,cf);
    dz = [];
  end
  for j=1:Q
    par.g{j} = stochastic_update(par.g{j},cf,dloghyp{j},[],dz{j});
  end
  
  % update dw
  testdw = zeros(P,Q);
  for i=1:P
    for j=1:Q
      testdw(i,j) = dw{j}(i);
    end
  end
  dw = reshape(cell2mat(dw),P,Q);
  assert(all(all(testdw == dw)));
  par.delta_w = cf.momentum_w*par.delta_w + cf.lrate_w*dw;
  par.w = par.w + par.delta_w;
  
  % update dbeta
  par.delta_beta = cf.momentum * par.delta_beta + cf.lrate_beta * dbeta;
  par.beta = par.beta + par.delta_beta;

  % update h_i
  wAm = cell(P,1); % wAm{i} = \sum_j w_ij A_j(oi) m_j
  for j=1:Q
    Aj = computeKnmKmminv(cf.covfunc_g, par.g{j}.loghyp, x, par.g{j}.z);
    for i=1:P
      indice = par.idx(:,i);
      w = par.w(i,j);
      if isempty(wAm{i})
        wAm{i} = w*Aj(indice,:)*par.g{j}.m;
      else
        wAm{i} = wAm{i} + w*Aj(indice,:)*par.g{j}.m;
      end
    end
  end
  for i=1:P
    indice = par.idx(:,i);
    y0 = ybatch(indice,i) - wAm{i};
    par.task{i}.beta = par.beta(i);
    % variational parameters of h_i
    par.task{i} = svi_update(xbatch(indice,:),y0,par.task{i},cf,cf.covfunc_h);
    % hyperparameters of h_i
    if cf.learn_z
      [~,dloghyp,~,dz] = svi_elbo(xbatch(indice,:),y0,par.task{i},cf.covfunc_h);
    else
      [~,dloghyp,~] = svi_elbo(xbatch(indice,:),y0,par.task{i},cf.covfunc_h);
      dz = [];
    end
    par.task{i} = stochastic_update(par.task{i},cf,dloghyp,[],dz);
  end
  
  % elbo for checking convergence
  par.idx = observed;
  elbo(iter) = slfm_elbo(x,y,par,cf);
  fprintf('Iteration \t%d: %.4f\n', iter, elbo(iter));
end
end

