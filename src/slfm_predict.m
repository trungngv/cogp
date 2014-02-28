function [mu, var, mu_g, var_g] = slfm_predict(covfunc_g, covfunc_h, par, xstar)
%SLFM_PREDCIT [mu var, mu_g, var_g] = slfm_predict(covfunc_g, covfunc_h, params, xstar)
%   Makes predictions with ssvi.
%
P = numel(par.task); Q = numel(par.g);
mu_g = zeros(size(xstar,1),Q);
var_g = mu_g;
for j=1:Q
  g = par.g{j};
  [mu_g(:,j), var_g(:,j)] = gpsvi_predict(covfunc_g, g.loghyp, g.m, g.S, g.z, xstar);
end
mu = zeros(size(xstar,1),P);
var = mu;
for i=1:P
  h = par.task{i};
  [mu(:,i), var(:,i)] = gpsvi_predict(covfunc_h, h.loghyp, h.m, h.S, h.z, xstar);
end
for i=1:P
  for j=1:Q
    w = par.w(i,j);
    mu(:,i) = w*mu_g(:,j) + mu(:,i);
    var(:,i) = w*w*var_g(:,j) + var(:,i);
  end
end

