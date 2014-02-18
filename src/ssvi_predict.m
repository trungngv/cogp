function [mu var, mu_g, var_g] = ssvi_predict(covfunc_g, covfunc_h, params, xstar)
%SSVI_PREDCIT [mu var, mu_g, var_g] = ssvi_predict(covfunc_g, covfunc_h, params, xstar)
%   Makes predictions with ssvi.
%
g = params.g;
[mu_g, var_g] = gpsvi_predict(covfunc_g, g.loghyp, g.m, g.S, g.z, xstar);
P = numel(params.task);
mu = zeros(size(xstar,1),P);
var = mu;
for i=1:P
  h = params.task{i};
  [mu(:,i), var(:,i)] = gpsvi_predict(covfunc_h, h.loghyp, h.m, h.S, h.z, xstar);
end
mu_g = repmat(mu_g,1,P);
var_g = repmat(var_g,1,P);
mu = mu_g + mu;
var = var_g + var;

