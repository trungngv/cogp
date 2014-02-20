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
  %params.task{i} = init_params(x(observed(:,i),:),y(observed(:,i),i),M,nhyper,0,[]);
  params.task{i} = init_params(x(observed(:,i),:),y(observed(:,i),i),M,nhyper_h,0,[]);
  params.task{i}.m = zeros(size(params.task{i}.m));
%  params.task{i}.S = diag(1*ones(numel(params.task{i}.m),1));
end
params.w = ones(P,1);
elbo = zeros(numel(cf.maxiter),1);
for i = 1 : cf.maxiter
  idx = randperm(N, cf.nbatch);
  params.idx = ~isnan(y(idx,:));
  params = ssvi_update(x(idx,:),y(idx,:),params,cf,i);
  params.idx = observed;
  elbo(i) = ssvi_elbo(x,y,params,cf);
end
 
% disp('initial hyp .vs learned hyp (including sigma2n)')
% fprintf('%.5f\t%.5f\n', [[loghyp;1/beta0],[params.loghyp;1/params.beta]]');

end

