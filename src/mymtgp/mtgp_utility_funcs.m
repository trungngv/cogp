% Generate functions by the factorized kernel.
%
% Author: Trung V. Nguyen
% Last update: 30/07/2012

function mtgp_utility_funcs()
% D: input dimensions, Q : num users, N : num items
D = 1; Q = 1; N = 400;
X = linspace(-2, 2, N)';
hyp = log(rand(D + 1));

Kf = 1;
Kx = covSEard(hyp, X);
K = kron(Kf, Kx);
  
% Sampling from Gaussian N(0, K) gives f = [f1 ... fQ]
%rng(30, 'twister');
Kchol = jit_chol(K)';
Z = randn(N*Q,1);
fqx = Kchol*Z;
fq = reshape(fqx, N, Q)';
  
% plot the utility functions of all Q users
figure(4); hold on;
plot(X, fq(1,:), '.k', 'LineWidth', 2, 'MarkerSize', 5);
title('MTGP utility functions');

end

