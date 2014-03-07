function [elbo,par] = slfm2_learn(x,y,M,par,cf)
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
P = size(y,2); Q = numel(par.g);
par.w = cf.w;
par.delta_w = zeros(P,Q);
par.beta = cf.beta*ones(P,1);
par.delta_beta = zeros(P,1);
observed = ~isnan(y);
nhyper_g = eval(feval(cf.covfunc_g));
ytmp = y;
for i=1:P
  yi = y(observed(:,i),i);
  ytmp(~observed(:,i),i) = mean(yi);
end
ymean = mean(ytmp,2);
for j=1:Q
  par.g{j} = init_params(x,ymean,M,nhyper_g,initz(x,M,cf.initz));
  par.g{j}.m = zeros(size(par.g{j}.m));
  if strcmp(cf.initz,'espaced')
    par.g{j}.m = par.g{j}.m + 1e-4*randn(M,1); % small pertubance to break symmetry
  end
end
% par.g{1}.loghyp = log([0.5; 1]);
% par.g{2}.loghyp = log(ones(size(par.g{2}.loghyp)));
elbo = zeros(cf.maxiter,1);
for iter = 1:cf.maxiter
  idx = randperm(N, cf.nbatch);
  par.idx = ~isnan(y(idx,:));
  xbatch = x(idx,:); ybatch = y(idx,:);
  
  % variational parameters of g_j
  par = slfm2_update_g(xbatch,ybatch,par,cf);
  % fix covariance hyperparameter in the first eposch
  if iter > N/cf.nbatch || ~cf.fix_first 
    % hyperparameters of g_j
    [~,dbeta,dw,dloghyp] = slfm2_elbo(xbatch,ybatch,par,cf);
    for j=1:Q
      [par.g{j}.loghyp,par.g{j}.delta_hyp] = stochastic_update(...
        par.g{j}.loghyp,par.g{j}.delta_hyp,dloghyp{j},cf.momentum,cf.lrate_hyp);
    end

    % update w, beta
    dw = reshape(cell2mat(dw),P,Q);
    [par.w,par.delta_w] = stochastic_update(par.w,par.delta_w,dw,cf.momentum_w,cf.lrate_w);
    [par.beta,par.delta_beta] = stochastic_update(par.beta,par.delta_beta,dbeta,cf.momentum,cf.lrate_beta);
    
    % update inducing of g (using new hyperparameters)
    if cf.learn_z
      [~,~,~,~,dz] = slfm2_elbo(xbatch,ybatch,par,cf);
      for j=1:Q
        [par.g{j}.z,par.g{j}.delta_z] = stochastic_update(...
          par.g{j}.z,par.g{j}.delta_z,dz{j},cf.momentum_z,cf.lrate_z);
      end
    end
  end
  
  % global elbo for checking convergence
  if ~isempty(cf.monitor_elbo) && mod(iter,cf.monitor_elbo) == 0
    nn = ceil(N/2000);
    for i=1:nn-1
      indice = ((i-1)*2000+1):(i*2000);
      par.idx = ~isnan(y(indice,:));
      elbo(iter) = elbo(iter) + slfm2_elbo(x(indice,:),y(indice,:),par,cf);
    end
    indice = ((nn-1)*2000+1):N;
    par.idx = ~isnan(y(indice,:));
    elbo(iter) = elbo(iter) + slfm2_elbo(x(indice,:),y(indice,:),par,cf);
    fprintf('Iteration\t%d:\t%.4f\n',iter,elbo(iter));
  else
    disp(['iter ' num2str(iter)]);
  end
  
end
end

