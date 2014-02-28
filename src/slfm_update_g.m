function par = slfm_update_g(x,y,par,cf)
%SLFM_UPDATE_G params = slfm_update_g(x,y,par,cf)
%   
% Update the variational parameters of the shared processes g.
%
% INPUT
%   - x : inputs (in SVI, this is a mini-batch)
%   - y : outputs (of inputs x)
%   - par
%   - cf : options for the optimization procedure
%
% OUTPUT
%   - params : updated parameters
%
% Trung V. Nguyen
% 27/02/14
P = numel(par.task); Q = numel(par.g);
wAm = cell(P,1); % wAm{i} = \sum_j w_ij A_j(oi) m_j
Ag = cell(Q,1);
Kmminv = cell(Q,1);
for j=1:Q
  [Ag{j},~,Kmminv{j}] = computeKnmKmminv(cf.covfunc_g, par.g{j}.loghyp, x, par.g{j}.z);
  for i=1:P
    indice = par.idx(:,i);
    w = par.w(i,j);
    if isempty(wAm{i})
      wAm{i} = w*Ag{j}(indice,:)*par.g{j}.m;
    else
      wAm{i} = wAm{i} + w*Ag{j}(indice,:)*par.g{j}.m;
    end
  end
end

for j=1:Q
  Lambda = Kmminv{j};
  tmp = zeros(size(par.g{j}.m));
  for i=1:P
    w = par.w(i,j); w2 = w*w;
    betaval = par.task{i}.beta;
    indice = par.idx(:,i);
    Lambda = Lambda + betaval*w2*(Ag{j}(indice,:)')*Ag{j}(indice,:);
    Ai = computeKnmKmminv(cf.covfunc_h, par.task{i}.loghyp, x(indice,:), par.task{i}.z);
    y0 = y(indice,i) - Ai*par.task{i}.m - (wAm{i} - w*Ag{j}(indice,:)*par.g{j}.m);
    tmp = tmp + betaval*w*Ag{j}(indice,:)'*y0;
  end

  Sinv = invChol(jit_chol(par.g{j}.S));
  theta1_old = Sinv*par.g{j}.m;
  theta2_old = -0.5*Sinv;
  theta2 = theta2_old + cf.lrate*(-0.5*Lambda - theta2_old);
  theta1 = theta1_old + cf.lrate*(tmp - theta1_old);
  [VV DD] = eig(theta2);
  invTheta2 = VV*diag(1./diag(DD))*VV';
  par.g{j}.S = - 0.5*invTheta2;
  par.g{j}.m = par.g{j}.S*theta1;
end

