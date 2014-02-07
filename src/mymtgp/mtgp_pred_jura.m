function mtgp_pred_jura()
% function jura_pred_mtgp()
% 1. Load data from Jura swiss data set
% 2. Sets parameters for learning (including initialization) 
% 3. Learns hyperparameters with minimize function
% 4. Makes predictions at all points on all tasks
% 5. Plots the predictions
%%
% rand('state',18);
% randn('state',20);
M = 3;    % 3 tasks, one primary and two secondary
D = 2;    % input space = spatial coordinates of locations (x, y)
covfunc_x = 'covSEard'; % Covariance function on input space

% Load data from Jura swiss data set
[XTemp, yTemp, XTestTemp, yTestTemp] = mapLoadData(['juraData' 'Cd']);
x = XTemp{2};
[N D]= size(x); % N = number of sample per tasks
n = N*M;        % n = N * M = total number of output points
% N x M matrix where each row is the output of M tasks
Y = [[yTemp{1}; yTestTemp{1}], yTemp{2}(1:N), yTemp{3}(1:N)];
%Y = [yTemp{1}(1:N), yTemp{2}(1:N), yTemp{3}(1:N)];
ymean = zeros(1,M);
ystd = zeros(1,M);
%NOTE: Y(260:end,1) not in smale scale, but it should not be used
%[Y(1:259,1) ymean(1) ystd(1)] = standardize(log(Y(1:259,1)),1,[],[]);
[Y(:,1) ymean(1) ystd(1)] = standardize(log(Y(:,1)),1,[],[]);
[Y(:,2:3) ymean(2:3) ystd(2:3)] = standardize(log(Y(:,2:3)),1,[],[]);
Y(260:end,1) = nan;
y = Y(:);
v = repmat((1:M),N,1);
ind_kf = v(:); % indices to tasks
v = repmat((1:N)',1,M);
ind_kx = v(:); % indices to input space data-points

%%
% 2. Set up data points for training
nx = ones(n-100,1); % Number of observations on each task-input point
xtrain = x;
ytrain = y;
ind_kx_train = ind_kx;
ind_kf_train = ind_kf;

ytrain = y(~isnan(y));
Y_isnan = isnan(Y); Y_isnan = Y_isnan(:);
ind_kf_train = ind_kf(~Y_isnan);
Y_isnan = isnan(Y'); Y_isnan = Y_isnan(:);
ind_kx_train = ind_kx(~Y_isnan);

% 3. Plotting all data and training data here
plot3(x(:,1),x(:,2),Y,'ko','markersize',7);
hold on;
for j = 1 : M 
  idx = find(ind_kf_train == j);
  plot3(x(ind_kx_train(idx),1),x(ind_kx_train(idx),2),...
        Y(ind_kx_train(idx),j),'k.','markersize',18);
  hold on;
end

% 4. Settings for learning
irank = 2;                         % Full rank

% Notice that Kf is a positive semi-definite matrix that specifies the
% inter-task similaries. Here Kf is parameterized with M*(M+1)/2
% parameters. Had we use the same kernel function, such as gaussian, with
% same scale parameter then number of parameter would be 1. On the other
% hand, if use different scales for each pair of fq, fq' then we need the
% same numbers to parametrize Kf; hence can learn the elements in Kf
% directly rather than learning the scales.

nlf = irank*(2*M - irank +1)/2;    % Number of parameters for Lf
theta_lf0 = init_Kf(M,irank);      % params for L, the cholesky decomposition of Kf
theta_kx0 =  init_kx(D);           % params for Kx
theta_sigma0 = init_sigma(M);      % noise variance
logtheta_all0 = [theta_lf0; theta_kx0; theta_sigma0];

% 5. We learn the hyper-parameters here
%    Optimize wrt all parameters except signal variance as full variance
%    is explained by Kf
% ignore sf2 (constant and do not learn it) at position nlf+1
deriv_range = ([1:nlf,nlf+2:length(logtheta_all0)])'; 
logtheta0 = logtheta_all0(deriv_range);
niter = 100; % setting for minimize function: number of function evaluations
[logtheta nl] = minimize(logtheta0,'learn_mtgp',niter, logtheta_all0, ...
			 covfunc_x, xtrain, ytrain,...
			 M, irank, nx, ind_kf_train, ind_kx_train, deriv_range);
%
% Update whole vector of parameters with learned ones
logtheta_all0(deriv_range) = logtheta;
theta_kf_learnt = logtheta_all0(1:nlf);
Ltheta_x = eval(feval('covSEard')); % Number of parameters of theta_x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% original code
%theta_kx_learnt = logtheta_all0(nlf+1 : nlf+Ltheta_x)
% modified code
theta_kx_learnt = [logtheta_all0(nlf+2:nlf+Ltheta_x); 0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta_sigma_learnt = logtheta_all0(nlf+Ltheta_x+1:end);

%6. Prediction
for task=1:M
  xtest = XTestTemp{task};
  Ntest = size(xtest,1);
  Ymean = repmat(ymean, size(xtest,1),1);
  Ystd = repmat(ystd, size(xtest,1),1);
  [alpha, Kf, L, Kxstar] = alpha_mtgp(logtheta_all0, covfunc_x, xtrain, ytrain,...
				    M, irank, nx, ind_kf_train, ...
				    ind_kx_train, xtest);
  all_Kxstar = Kxstar(ind_kx_train,:);
  Kf_task = Kf(ind_kf_train,task);
  Ypred = (repmat(Kf_task,1,Ntest).*all_Kxstar)'*alpha;
  Ypred = exp(Ypred .* Ystd(:,task) + Ymean(:,task));
  disp(['Mean absoluate error -- task ' num2str(task)])
  mean(abs((yTestTemp{task} - Ypred)))
end

% 7. Plotting the negative marginal log-likelihood
hold off;
figure(task), plot(nl); title('Negative Marginal log-likelihood');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta_kx0 = init_kx(D)
theta_kx0 = [log(1); log(rand(D,1))];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta_lf0 = init_Kf(M,irank)
Kf0 = eye(M);                 % Init to diagonal matrix (No task correlations)
Lf0 = chol(Kf0)';
theta_lf0 = lowtri2vec_inchol(Lf0,M,irank);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta_sigma0 = init_sigma(M)
theta_sigma0 =  (1e-7)*rand(M,1);  
