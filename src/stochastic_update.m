function params = stochastic_update(params,cf,dloghyp,dbeta,dz)
%STOCHASTIC_UPDATE params = stochastic_update(params,cf,dloghyp,dbeta,dz)
%   Stochastic update of the parameters.

params.delta_hyp = cf.momentum * params.delta_hyp + cf.lrate_hyp * dloghyp;
params.loghyp = params.loghyp + params.delta_hyp;
if ~isempty(dbeta)
  params.delta_beta = cf.momentum * params.delta_beta + cf.lrate_beta * dbeta;
  params.beta = params.beta + params.delta_beta;
end
if ~isempty(dz)
  params.delta_z = cf.momentum_z*params.delta_z + cf.lrate_z * dz;
  params.z = params.z + params.delta_z;
end

