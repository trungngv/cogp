% Sampling utility functions using the multi-output dependent gaussian
% processes (in Neil's paper)
function cgp_utility_funcs()

  % Construct the covariance matrix K between all points
  %rng(0, 'twister'); % to keep the GP parameters consistent between runs
  D = 1; % input dimensions
  R = 2; % latent functions
  Q = 2; % number of users
  N = length(-2:0.005:2); % number of items
  axes = [-2, 2, -3, 3];
  % Set S = Q x R for each of the smoothing kernel k_qr
  S = 5 + 10 .* abs(randn(Q, R)) % each S_qr ~ N(5, 10)
  % Same setting for S as in the paper (dependent outputs)
  % S = [1; 2; 5; 10];
  % Setting for independent outputs (does it? still share the same latent
  % function r)
  %S(1, 1) = 1; S(2, 1) = 2; S(3, 1) = 5; S(4, 1) = 10;
  
  % Set L{qr} for each smoothing kernel function k_qr
  LQR = cell(Q, R);
  for u=1:Q
    for r=1:R
      % Trung: Set similar LQR{q, r} to get similar utility functions
      lower = sampleLowerDiagonal(D);
      % Use cholesky decomposition to get a pd matrix
      LQR{u, r} = lower * lower';
      % Use diagonal LQR{q, r} for testing with multi-gp
      LQR{u, r} = LQR{u, r} .* eye(D)
    end
  end
  
  % Set 1, 2 to be similar and 3, 4 to be similar; otherwise different
%   for r=1:R
%     LQR{2, r} = LQR{1, r} + normrnd(0, 3); % add noise 1
%     LQR{3, r} = LQR{3, r} .* abs(randn(D, D)); % make 1 and 3 different
%     LQR{4, r} = LQR{3, r} + normrnd(0, 3); 
%   end

  % Independent setting
  %LQR{1, 1} = 10, LQR{2, 1} = 100, LQR{3, 1} = 200, LQR{4, 1} = 300;
  % Same setting as in the paper
%   LQR{1, 1} = 50;
%   LQR{2, 1} = 50;
%   LQR{3, 1} = 300;
%   LQR{4, 1} = 200;
  
  % Set the length-scales L_r for covariance functions of the latent
  % processes (currently using same length-scale for all dimensions).
  L = cell(1, R);
  for r=1:R
    %L{r} = [L unifrnd(1, 200) * eye(D)];
    L{r} = (50 + unifrnd(0, 200)) * eye(D)
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Parameters for experiment to generate 2 dependent outputs (D = 1)
%   S = zeros(Q, 4);
%   S = [10 7 10 5; ...
%        11 8 9 4;...
%        2  3 8 1;...
%        15 6 1 1];
%   % LQR{q,r}: DxD :
%   LQR = cell(Q, 4);
%   LQR{1,1} = 200; LQR{2,1} = 250; 
%   LQR{1,2} = 400; LQR{2,2} = 450;
%   LQR{1,3} = 508; LQR{2,3} = 100;
%   LQR{1,4} = 412; LQR{2,4} = 100;
%   % user 3 and 4
%   LQR{3,1} = 600; LQR{4,1} = 100; 
%   LQR{3,2} = 600; LQR{4,2} = 100;
%   LQR{3,3} = 208; LQR{4,3} = 100;
%   LQR{3,4} = 212; LQR{4,4} = 100;
%   
%   % L{r} : DxD - latent functions we want; small => large step-length,
%   % constant func; large => small step-length, wavy latent
%   L = cell(1,4);
%   L{1} = 645;
%   L{2} = 401;
%   L{3} = 1000;
%   L{4} = 300;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Parameters for experiment to generate 2 independent outputs (D = 1)
%   S = zeros(Q, 4);
%   S = [15 15 10 5; ...
%        5 2 8 1;...
%        11 8 9 4;...
%        15 6 1 1];
%   % LQR{q,r}: DxD :
%   LQR = cell(Q, 4);
%   LQR{1,1} = 500; LQR{1,2} = 50;
%   LQR{2,1} = 50; LQR{2,2} = 200;
%   
%   % L{r} : DxD - latent functions we want; small => large step-length,
%   % constant func; large => small step-length, wavy latent
%   L = cell(1,4);
%   L{1} = 400;
%   L{2} = 309;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Parameters learned from the LearnParametersCGP function (for experiment2)
% S(1,1) = 1.3111;
% S(1,2) = 1.3111;
% S(2,1) = 1.9703;
% S(2,2) = 1.9703;
% S(3,1) = -0.5291;
% S(3,2) = -0.5291;
% L{1} = 66.3245;
% L{2} = 66.3245;
% LQR{1,1} = 23.7208;
% LQR{1,2} = 23.7208;
% LQR{2,1} = 3.3417;
% LQR{2,2} = 3.3417;
% LQR{3,1} = 70.4088;
% LQR{3,2} = 70.4088;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters learned from the LearnParametersCGP function (for experiment3)
L{1} = 79.9856;
L{2} = L{1};
S(1,1) = 0.6776;
S(1,2) = 0.6776;
S(2,1) = 0.6776;
S(2,2) = 0.6776;
LQR{1,1} = 43.9995;
LQR{1,2} = 43.9995;
LQR{2,1} = 43.9995;
LQR{2,2} = 43.9995;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
  % Generate random item features (input points)
  rngSeed = 230;
  rngType = 'twister';
  rng(rngSeed, rngType);
  items = (-2:0.005:2)';
  
  % save all data for this experiment
%   file = ['convolvedGPParams-latents' num2str(R) '-seed' num2str(rngSeed) '.mat'];
%   save(file, 'S', 'LQR', 'L', 'rngSeed', 'rngType');
%   disp(['Saved data to file ' file]);
  
  % initialize the cumulative covariance matrix
  cumMatrix = cell(Q);
  for q1=1:Q
    for q2=1:Q
      cumMatrix{q1, q2} = zeros(N, N);
    end
  end
  
  % Compute covariance for each latent process and sum them up
  for r=1:R
    covr = constructCovMatrix(r);
    for q1=1:Q
      for q2=1:Q
        cumMatrix{q1, q2} = cumMatrix{q1, q2} + covr{q1, q2};
      end
    end
  end
  
  % The covariance matrix K(q,x,q',x') of size QN * QN
  % where first row is the covariance b.w q=1,x=1 and all others:
  % [1,1,q=1,x=1 ... 1,1,q=1,x=N, 1,1,q=2,x=1, ...,
  % 1,1,q=2,x=N, ..., 1,1,q=Q,x=1, ... 1,1,q=Q,x=N]
  % (must consist of Q*N elements)!
  
  % Sampling from Gaussian N(0, K) gives an utility function 
  
  % TRUNG: the distribution f_1(x) by generateConvolvedFunctions # the
  % marginal function generated here because of the sequences of number
  % generated are different.
  K = cell2mat(cumMatrix);
  fqx = gsamp(zeros(size(K, 1), 1), K, 1); % fqx has size 1xQN (1 row)
  % Reshape fqx to get Q utility functions for Q users
  fq = reshape(fqx, N, Q)';
  % Note this only works for 1-D
%   trainingIdx = find(items <= 0.75 * 2);
%   trainingItems = items(trainingIdx,:);
%   trainingFq = fq(:, trainingIdx);
%   save('temp.mat', 'trainingItems', 'trainingFq');
  %save('temp.mat', 'items', 'fq');
  
  % plot the utility functions of all Q users (1-D)
  figure(1); hold on;
  color = ['m', 'g', 'b', 'k'];
  for q=1:Q
    plot(items, fq(q, :), ['.-' color(q)], 'MarkerSize', 2);
  end
  legend(num2str((1:Q)'));
  title(['Utility functions (convolved GP) for ' num2str(R) ' latent funcs']);
  axis(axes);

  % plot the utility functions of all Q users (2-D)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Sample a lower diagonal matrix with dimension D and positive diagonal
  function lower = sampleLowerDiagonal(D)
    lower = zeros(D, D);
    for i = 1:D
      lower(i, i) = 5 + abs(normrnd(3, 10));
    end
    for i = 2:D
      for j = 1:i-1
        % mean = 4, std = 4
        lower(i, j) = 4 + abs(3 * normrnd(1, 7));
      end
    end
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % construct the covariance matrix for the convolved GP process by the
  % r_th latent function
  function C = constructCovMatrix(r)
    % S : the Q x 1 matrix, cofficients of the smoothing kernels k_qr
    % [S_1r; S_2r; ...; S_Qr]
    % L : Q x D x D cell where each cell is the scales of the smoothing kernels k_qr
    % (D = % dimension of input, i.e., cardinality of feature vector of an item x)
    % L_r : the D x D matrix, scales of the kernel of the latent functions
    C = cell(Q);
    % Cannot use kronecker product because kernel is not separated into
    % products of kernel of q and kernel of x. But may use ind to do
    % multiplication (like Edwin did)
    for q=1:Q
      for s=1:Q
        C{q, s} = zeros(N, N);
        P = pinv(LQR{q, r}) + pinv(LQR{s, r}) + pinv(L{r});
        partition = S(q, r) * S(s, r) * (det(L{r})^-0.5) / sqrt(det(P));
        C{q,s} = feval('covSEard', [log(sqrt(P)); log(sqrt(partition))], items);
      end
    end
  end  

end
