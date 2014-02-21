function [elbo,params] = ssvi_learn(x,y,M,cf,z0)
%SSVI_LEARN ssvi_learn(x,y,M,cf,z0)
%   Quick wrapper to run structured GPSVI.
% 
% INPUT
%   - x,y : train inputs, train outputs
%   - M : #inducing inputs
%   - cf : configuration
%
% OUTPUT
%   - mu, s2 : mean and variance prediction
%   - elbo : the bound
%   - params : learned parameters
[N D] = size(x);
% get observation indice for each output
P = size(y,2);
ytmp = y;
observed = ~isnan(y);
for i=1:P
  yi = y(observed(:,i),i);
  ytmp(~observed(:,i),i) = mean(yi);
end
% for each input, the mean of all outputs are used to initialize m_g
ymean = mean(ytmp,2);
nhyper_g = eval(feval(cf.covfunc_g));
nhyper_h = eval(feval(cf.covfunc_h));
params.g = init_params(x,ymean,M,nhyper_g,0,[]);
params.g.m = zeros(size(params.g.m));
params.g.beta = []; params.g.delta_beta = [];
for i=1:P
  %params.task{i} = init_params(x(observed(:,i),:),y(observed(:,i),i),M,nhyper_h,0,[]);
  % this allows inducing inputs to appear in even unobserved region
  yi = y(:,i); yi(isnan(yi)) = 0;
  params.task{i} = init_params(x,yi,M,nhyper_h,0,[]);
  params.task{i}.m = zeros(size(params.task{i}.m));
%  params.task{i}.S = diag(1*ones(numel(params.task{i}.m),1));
end
params.w = ones(P,1);
params.delta_w = zeros(P,1);
elbo = zeros(numel(cf.maxiter),1);
for iter = 1:cf.maxiter
  idx = randperm(N, cf.nbatch);
  params.idx = ~isnan(y(idx,:));
  xbatch = x(idx,:); ybatch = y(idx,:);
  % variational parameters of g
  [params,A,Knm,Kmminv,Lmm] = ssvi_update_g(xbatch,ybatch,params,cf);
  % variational of h_i 
  A_g = computeKnmKmminv(cf.covfunc_g, params.g.loghyp, xbatch, params.g.z);
  for i=1:P
    indice = params.idx(:,i);
    w = params.w(i);
    y_minus_g = ybatch(indice,i) - w*A_g(indice,:)*params.g.m;
    params.task{i}= svi_update(xbatch(indice,:),y_minus_g,params.task{i},cf,cf.covfunc_h);
  end
  
  % hyperparameters of g
  if cf.learn_z
    [~,dloghyp,dw,dz] = ssvi_elbo(xbatch,ybatch,params,cf,A,Knm,Kmminv,Lmm);
    params.g = stochastic_update(params.g,cf,dloghyp,[],dz);
  else
    [~,dloghyp,dw] = ssvi_elbo(xbatch,ybatch,params,cf,A,Knm,Kmminv,Lmm);
    params.g = stochastic_update(params.g,cf,dloghyp,[],[]);
  end
  % update dw
  params.delta_w = cf.momentum_w*params.delta_w + cf.lrate_w*dw;
  params.w = params.w + params.delta_w;

  % update h_i
  % TODO: can also delay update of h_i until g is learned
  diagKnn = feval(cf.covfunc_g, params.g.loghyp, xbatch, 'diag');
  % TODO: if update loghyp_g and loghyp_i simultaneously, need to compute
  % A_g, Knm_g using the OLD loghyp_g.
  [A_g,Knm_g] = computeKnmKmminv(cf.covfunc_g, params.g.loghyp, xbatch, params.g.z);
  for i=1:P
    indice = params.idx(:,i);
    w = params.w(i); w2 = w*w;
    y_minus_g = ybatch(indice,i) - w*A_g(indice,:)*params.g.m;
    % hyperparameters of h_i
    if cf.learn_z
      [~,dloghyp,dbeta,dz] = svi_elbo(xbatch(indice,:),y_minus_g,params.task{i},cf.covfunc_h);
    else
      [~,dloghyp,dbeta] = svi_elbo(xbatch(indice,:),y_minus_g,params.task{i},cf.covfunc_h);
    end

%     this saves some computation but updates variational parameters of h_i AFTER hypers of g_i    
%     % variational parameters of h_i
%     [params.task{i},A,Knm,Kmminv,Lmm]= svi_update(xbatch(indice,:),y_minus_g,params.task{i},cf,cf.covfunc_h);
%     % hyperparameters of h_i
%     if cf.learn_z
%       [~,dloghyp,dbeta,dz] = svi_elbo(xbatch(indice,:),y_minus_g,params.task{i},cf.covfunc_h,A,Knm,Kmminv,Lmm);
%     else
%       [~,dloghyp,dbeta] = svi_elbo(xbatch(indice,:),y_minus_g,params.task{i},cf.covfunc_h,A,Knm,Kmminv,Lmm);
%     end
    
    % gradient of beta_i
    dbeta = dbeta -0.5*w2*sum(diagKnn(indice)-diagProd(A_g(indice,:),Knm_g(indice,:)'));
    dbeta = dbeta - 0.5*w2*traceABsym(params.g.S,A_g(indice,:)'*A_g(indice,:));
    if cf.learn_z
      params.task{i} = stochastic_update(params.task{i},cf,dloghyp,dbeta,dz);
    else
      params.task{i} = stochastic_update(params.task{i},cf,dloghyp,dbeta,[]);
    end
  end
  
  % elbo for checking convergence
  params.idx = observed;
  elbo(iter) = ssvi_elbo(x,y,params,cf);
end
 
% disp('initial hyp .vs learned hyp (including sigma2n)')
% fprintf('%.5f\t%.5f\n', [[loghyp;1/beta0],[params.loghyp;1/params.beta]]');

end

