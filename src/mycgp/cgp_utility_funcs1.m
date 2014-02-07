% Experiment 1: Utilities functions that can be generated with different
% number of latent functions using the convolved GPs.

% Trung Nguyen
% May 2012

function cgp_utility_funcs()

  rng(0, 'twister'); % to keep the GP parameters consistent between runs
  D = 1; % input dimensions
  R = 1; % latent functions
  Q = 1; % number of users
  N = 400; % number of items

  % S_qr: s mall : constant function, s large: curvy function
  S = zeros(Q, 4);
  S(1,1) = 15;
  S(1,2) = 5;
  S(1,3) = 10;
  S(1,4) = 5;
  % LQR{q,r}: DxD :
  LQR = cell(Q, 4);
  LQR{1,1} = 500;
  LQR{1,2} = 300;
  LQR{1,3} = 508;
  LQR{1,4} = 412;
  % L{r} : DxD - latent functions we want; small => large step-length,
  % constant func; large => small step-length, wavy latent
  % small length-step such as 50 is suitable for utility function
  L = cell(1,4);
  L{1} = 59;
  L{2} = 500;
  L{3} = 230;
  L{4} = 310;
  
  % Generate random item features (input points)
  items = zeros(N, D);
  rngSeed = 30;
  rngType = 'twister';
  rng(rngSeed, rngType);
  for itemIdx=1:N
    items(itemIdx, :) = rand(1, D); % each feature ~ N(0, 1)
  end
  
  % initialize the cumulative covariance matrix
  cumMatrix = cell(Q);
  cumMatrix{1,1} = zeros(N, N);
  
  % Compute covariance for each latent process and sum them up
  for r=1:R
    covr = constructCovMatrix(r);
    cumMatrix{1,1} = cumMatrix{1,1} + covr{1,1};
  end
  
  % The covariance matrix K(q,x,q',x') of size QN * QN
  % where first row is the covariance b.w q=1,x=1 and all others:
  % [1,1,q=1,x=1 ... 1,1,q=1,x=N, 1,1,q=2,x=1, ...,
  % 1,1,q=2,x=N, ..., 1,1,q=Q,x=1, ... 1,1,q=Q,x=N]
  % (must consist of Q*N elements)!
  
  % Sampling from Gaussian N(0, K) gives an utility function 
  K = cell2mat(cumMatrix);
  fqx = gsamp(zeros(size(K, 1), 1), K, 1); % fqx has size 1xQN (1 row)
  
  figure(3); hold on;
  plot(items, fqx, '.b', 'LineWidth', 2, 'MarkerSize', 5);
  title('Utility functions (convolved GP) for variable smoothing kernel');
  legend(['Length-steps Lr = ' num2str(50)], num2str(200), num2str(500))
  axis([0,1,-30,30]);
  
  % construct the covariance matrix for the convolved GP process by the
  % r_th latent function
  function C = constructCovMatrix(r)
    % S : the Q x 1 matrix, cofficients of the smoothing kernels k_qr
    % [S_1r; S_2r; ...; S_Qr]
    % L : Q x D x D cell where each cell is the scales of the smoothing kernels k_qr
    % (D = % dimension of input, i.e., cardinality of feature vector of an item x)
    % L_r : the D x D matrix, scales of the kernel of the latent functions
    C = cell(Q);
    C{1,1} = zeros(N,N);
    % Cannot use kronecker product because kernel is not separated into
    % products of kernel of q and kernel of x.
    P = pinv(LQR{1,r}) + pinv(LQR{1,r}) + pinv(L{r});
    partition = S(1,r) * S(1,r) * (det(L{r})^-0.5) / sqrt(det(P));
    Pinv = pinv(P);
    for i=1:N
      for j=1:N
        C{1,1}(i,j) = partition * exp(-0.5 * (items(i,:) - items(j,:)) * Pinv * (items(i,:)-items(j,:))');
      end
    end
  end
  
end