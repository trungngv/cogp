function [z,z0,hyp,mu,s2] = learn_fitc(x,y,xtest,M,z0)
%LEARN_FITC [z,z0,hyp,mu,s2] = learn_fitc(x,y,xtest,M,z0)
%   Wrapper for FITC.
% 
% INPUT
%   - x, y, xtest : train inputs, train outputs, test inputs (optional)
%   - M : #inducing points
%   - z0 : initial inducing inputs (optional)
%
% OUTPUT
%   - z, z0: learned and initial inducing inputs
%   - hyp : learned hyp
%   - mu : predictive mean 
%   - s2 : predictive variance

me_y = mean(y); y0 = y - me_y; % zero mean the data
[N,dim] = size(x);

% initialize pseudo-inputs to a random subset of training inputs
if isempty(z0)
  [dum,I] = sort(rand(N,1)); clear dum;
  I = I(1:M);
  z0 = x(I,:);
end

% initialize hyperparameters sensibly (see spgp_lik for how
% the hyperparameters are encoded)
% hyp_init(1:dim,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
% hyp_init(dim+1,1) = log(var(y0,1)); % log size 
% hyp_init(dim+2,1) = log(var(y0,1)/2); % log noise
hyp_init(1:dim+2,1) = log(ones(dim+2,1));

% optimize hyperparameters and pseudo-inputs
w_init = [reshape(z0,M*dim,1);hyp_init];
[w,f] = minimize(w_init,'spgp_lik',-1000,y0,x,M);
% [w,f] = lbfgs(w_init,'spgp_lik',200,10,y0,x,M); % an alternative
z = reshape(w(1:M*dim,1),M,dim);
hyp = w(M*dim+1:end,1);

% PREDICTION
if ~isempty(xtest)
  [mu0,s2] = spgp_pred(y0,x,z,xtest,hyp);
  mu = mu0 + me_y; % add the mean back on
  % if you want predictive variances to include noise variance add noise:
  %s2 = s2 + exp(hyp(end));
end  

return;

