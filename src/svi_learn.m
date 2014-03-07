function [mu,s2,elbo,par] = svi_learn(x,y,xtest,M,cf,z0)
%LEARN_GPSVI learn_gpsvi(x,y,xtest,M,cf,z0)
%   Quick wrapper to run GPSVI.
% 
% INPUT
%   - x,y,xtest : train inputs, train outputs, test outputs
%   - M : #inducing inputs
%   - cf : configuration
%
% OUTPUT
%   - mu, s2 : mean and variance prediction
%   - elbo : the bound
%   - params : learned parameters
[N D] = size(x);
nhyper = eval(feval(cf.covfunc));
if isempty(z0)
  par = init_params(x,y,M,nhyper,initz(x,M,cf.initz));
else
  par = init_params(x,y,M,nhyper,z0);
end
par.beta = cf.beta;
par.delta_beta = 0;
elbo = zeros(cf.maxiter,1);
for iter = 1 : cf.maxiter
  idx = randperm(N, cf.nbatch);
  xi = x(idx,:); yi = y(idx);
  [par,A,Knm,Kmminv,Lmm] = svi_update(xi,yi,par,cf,cf.covfunc);
  if iter > N/cf.nbatch   % fix covariance hyperparameter in the first eposch
    [~,dloghyp,dbeta] = svi_elbo(xi,yi,par,cf.covfunc,A,Knm,Kmminv,Lmm);
    [par.loghyp,par.delta_hyp] = stochastic_update(par.loghyp,par.delta_hyp,...
      dloghyp, cf.momentum, cf.lrate_hyp);
    [par.beta,par.delta_beta] = stochastic_update(par.beta,par.delta_beta,...
      dbeta, cf.momentum, cf.lrate_beta);
    if cf.learn_z
      [~,~,~,dz] = svi_elbo(xi,yi,par,cf.covfunc);
      [par.z,par.delta_z] = stochastic_update(par.z, par.delta_z, dz, ...
        cf.momentum_z, cf.lrate_z);
    end
  end
  % compute global elbo using updated parameters
  if ~isempty(cf.monitor_elbo) && mod(iter,cf.monitor_elbo) == 0
    nn = ceil(N/2000);
    for i=1:nn-1
      indice = ((i-1)*2000+1):(i*2000);
      elbo(iter) = elbo(iter) + svi_elbo(x(indice,:),y(indice),par,cf.covfunc,[],[],[],[]);
    end
    indice = ((nn-1)*2000+1):N;
    elbo(iter) = elbo(iter) + svi_elbo(x(indice,:),y(indice),par,cf.covfunc,[],[],[],[]);
    fprintf('Iteration\t%d:\t%.4f\n',iter,elbo(iter));
  else
    disp(['finished iter ' num2str(iter)]);
  end
end

if ~isempty(xtest)
  [mu s2] = gpsvi_predict(cf.covfunc, par.loghyp, par.m, par.S, par.z, xtest);
end

end

