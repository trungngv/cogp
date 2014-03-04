function [mu, var, mu_g, var_g] = slfm2_predict(covfunc_g, par, xstar, n_outputs)
%SLFM2_PREDCIT [mu var, mu_g, var_g] = slfm2_predict(covfunc_g, params, xstar, P)
%   Makes predictions with ssvi.
%
Q = numel(par.g);
mu_g = zeros(size(xstar,1),Q);
var_g = mu_g;
for j=1:Q
  g = par.g{j};
  [mu_g(:,j), var_g(:,j)] = gpsvi_predict(covfunc_g, g.loghyp, g.m, g.S, g.z, xstar);
end
mu = zeros(size(xstar,1),n_outputs);
var = mu;
for i=1:n_outputs
  for j=1:Q
    w = par.w(i,j);
    mu(:,i) = w*mu_g(:,j) + mu(:,i);
    var(:,i) = w*w*var_g(:,j) + var(:,i);
  end
end

