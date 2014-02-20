function params = ssvi_update(x,y,params,cf,iter)
%SSVI_UPDATE params = ssvi_update(x,y,params,cf,iter)
%   
% Update all parameters of the structured gp model.
% This update is similar to variational EM:
%   - first update the variational parameters of q(u_g) and q(u_i) (E-step)
%   - then update the hyperparameters of u_g and u_i (M-step)
%
% We can analyze the effect of the order of optimization: do we update
% q(u_g) and then its hyperparameters, or do we update q(u_g) and then
% q(u_i) and then all hyperparameters?
%
% If x is the entire training, this becomes the (natural) variational
% inference. 
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - params
%   - cf : options for the optimization procedure
%
% OUTPUT
%   - params : updated parameters
%
% Trung V. Nguyen
% 20/02/14
P = size(y,2);
% Update for g
[A,~,Kmminv] = computeKnmKmminv(cf.covfunc_g, params.g.loghyp, x, params.g.z);
Lambda = Kmminv;
tmp = zeros(size(params.g.m));
for i=1:P
  betaval = params.task{i}.beta;
  indice = params.idx(:,i);
  Lambda = Lambda + betaval*(A(indice,:)')*A(indice,:);
  Ai = computeKnmKmminv(cf.covfunc_h, params.task{i}.loghyp, x(indice,:), params.task{i}.z);
  y_minus_hi = y(indice,i) - Ai*params.task{i}.m;
  tmp = tmp + betaval*A(indice,:)'*y_minus_hi;
end

Sinv = invChol(jit_chol(params.g.S,4));
theta1_old = Sinv*params.g.m;
theta2_old = -0.5*Sinv;
theta2 = theta2_old + cf.lrate*(-0.5*Lambda - theta2_old);
theta1 = theta1_old + cf.lrate*(tmp - theta1_old);
[VV DD] = eig(theta2);
invTheta2 = VV*diag(1./diag(DD))*VV';
params.g.S = - 0.5*invTheta2;
params.g.m = params.g.S*theta1;
if cf.learn_z
  [~,dloghyp,dz] = ssvi_elbo(x,y,params,cf);
  params.g = stochastic_update(params.g,cf,dloghyp,[],dz);
else
  [~,dloghyp,~] = ssvi_elbo(x,y,params,cf);
  params.g = stochastic_update(params.g,cf,dloghyp,[],[]);
end

% Update h_i
diagKnn = feval(cf.covfunc_g, params.g.loghyp, x, 'diag');
if iter >= 0
  A = computeKnmKmminv(cf.covfunc_g, params.g.loghyp, x, params.g.z);
  for i=1:P
    indice = params.idx(:,i);
    y_discounted = y(indice,i) - A(indice,:)*params.g.m;
    % DON'T FORGET w_i
    params.task{i} = svi_update(x(indice,:),y_discounted,params.task{i},cf,cf.covfunc_h,[],[]);
    % contribution from g
    dbeta_g = -0.5*sum(diagKnn(indice)) - 0.5*traceABsym(params.g.S,A(indice,:)'*A(indice,:));
    params.task{i}.delta_beta = params.task{i}.delta_beta + cf.lrate_beta*dbeta_g;
    params.task{i}.beta = params.task{i}.beta + cf.lrate_beta*dbeta_g;
  end
end

end

