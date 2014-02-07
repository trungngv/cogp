% Generate utility functions using the factorized kernel (with approximation for the convolved GPs).

% Trung Nguyen
% April 2012
function mtgp()

  % Construct the covariance matrix K between all points
  D = 1; % input dimensions = 1
  Q = 2; % number of users
  N = 400; % number of items
  
  [C, Kf, items] = randomizeData(N, D);
  %file = 'convolvedGPParams-latents1-seed30.mat';
  %[S, LQR, L, items] = loadConvolvedGPParams(file, N, D);
  %[Kf, C, Cavg] = convolvedGPToMultiTaskGP(S, LQR, L);
  %[Kf, C, Cavg] = approximateConvolvedGP(S, LQR, L);
  
  % The covariance matrix K(q,x,q',x') of size QN * QN
  % where first row is [q=1,x=1,q=1,x=1 ... q=1,x=1,q=1,x=N, q=1,x=1,q=2,x=1, ...,
  % q=1,x=1,q=2,x=N, ..., q=1,x=1,q=Q,x=1, ... q=1,x=1,q=Q,x=N] (Q*N
  % elements)!
  % In other words, K(q,x,q',x') = kron(Kf, Kx)
  
  % Use Q^2 different covariance matrix
  %K = cell2mat(constructCovMatrix(Kf, C, items, items, true));
  
  % use the average Cavg
  %K = kron(Kf, gaussianCovariance(items, items, Cavg, true));
  
  % use randomized data
  K = kron(Kf, gaussianCovariance(items, items, C, true));
  
  % Sampling from Gaussian N(0, K) gives an utility function
  fqx = gsamp(zeros(size(K, 1), 1), K, 1); % fqx has size 1xQN (1 row)
  % Reshape fqx to get Q utility functions for Q users
  fq = reshape(fqx, N, Q)';
  
  % plot the utility functions of all Q users
  figure(2); hold on;
  color = ['r', 'y', 'b', 'k'];
  for q=1:Q
    plot(items, fq(q, :), ['.' color(q)], 'LineWidth', 2, 'MarkerSize', 5);
  end
  title('Utility functions (factorized kernel) (exact) for 2 latent funcs');
  legend(num2str([1; 2]));
  axis([0,1,-30,30]);
  
%   hold off;
%   for q=1:Q
%     figure(q);
%     d1 = items(:, 1);
%     d2 = items(:, 2);
%     f=fq(q, :);
%     [X, Y] = meshgrid(0:0.01:1,0:0.01:1);
%     Z = griddata(d1, d2, f, X, Y);
%     surf(X, Y, Z);
%   end
  
end

% Construct the covariance matrix between utlity functions using
% different C for each pair of outputs/tasks
% use isPrecision = true if the precision matrices are given
% and isPrecision = false if the covariance matrices are given
function K = constructCovMatrix(Kf, C, X, X2, isPrecision)
  Q = size(Kf, 1);
  K = cell(Q);
  % Cannot use kronecker product because kernel is not separated into
  % products of kernel of q and kernel of x.
  for q=1:Q
    for s=1:Q
      K{q,s} = Kf(q,s) * gaussianCovariance(X, X2, C{q, s}, isPrecision);
      if ~isPrecision
        K{q,s} = K{q,s} * (det(C{q, s})^-0.5); 
      else
        K{q,s} = K{q,s} * (det(C{q, s})^0.5);
      end
    end
  end
end

% construct a covariance matrix given a kernel function and two set of
% points
% TRUNG: this can be computed more efficiently using GPML package...
function C = gaussianCovariance(X, X2, L, isPrecision)
  nx = size(X, 1);
  nx2 = size(X2, 1);
  C = zeros(nx, nx2);
  if ~isPrecision
    prec = inv(L);
  else
    prec = L;
  end
  for i=1:nx
    for j=1:nx2
      C(i, j) = exp(-0.5 * (X(i,:) - X2(j,:)) * prec * (X(i,:) - X2(j,:))');
    end
  end
end

% Sample a lower diagonal matrix with dimension D and positive diagonal
function lower = sampleLowerDiagonal(D)
  lower = zeros(D, D);
  for i = 1:D
    lower(i, i) = 10 + normrnd(5, 5);
  end
  for i = 2:D
    for j = 1:i-1
      % mean = 5, std = 20
      lower(i, j) = abs(5 + 5 * randn(1, 1));
    end
  end
end

% Load all the parameters used in the convolved gp
function [S, LQR, L, items] = loadConvolvedGPParams(savedfile, N, D)
  load(savedfile);
  % This is critical to generate the same series of random numbers in both
  % program (convolved gp and mtgp)
  rng(rngSeed, rngType);
  items = rand(N, D);
end

% Generates randomized data for testing
function [C, Kf, items] = randomizeData(N, D)
  % Generate random item features (input points)
  rng(10, 'twister');
  items = rand(N, D);

  % length-scales matrix for kernel between items
  %L = diag(unifrnd(1, 500, D, 1));
  C = 100 * eye(D); % same scale for all dimension
  
  %lower = sampleLowerDiagonal(Q);
  %Kf = lower * lower';
  Kf = [[100 80]; [80 70]];
  %Kf = diag([10 40 80 100]) % use a toy similarity positive definite matrix
end
