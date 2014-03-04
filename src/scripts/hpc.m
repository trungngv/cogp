load hpc20000.csv
rng(1110, 'twister');
fmae = @(ytrue,ypred) mean(abs(ypred-ytrue(:,1)));
frmse = @(ytrue,ypred) sqrt(mean((ypred-ytrue(:,1)).^2));

x = hpc5000(:,3:end);
y = hpc5000(:,1:2);
features = [1,2,3,4,5];
ntest = 10000;
randind = randperm(size(x,1));
xtest = x(randind(1:ntest),features);
x = x(randind(ntest+1:end),features);
ytest = y(randind(1:ntest),:);
y = y(randind(ntest+1:end),:);

[x,xmean,xstd] = standardize(x,[],[]);
xtest = standardize(xtest,xmean,xstd);

showplots = false;

%% exploratory analysis
y1 = y(:,1); y2 = y(:,2);
disp('corr y1 vs y2')
corr(y1,y2)

if showplots
  for i=1:size(x,2)
    figure;
    plot(x(:,i),y1,'.'); 
    title(['x_' num2str(i) ' vs. y1']);
    disp('press to continue')
    pause
  end

  for i=1:size(x,2)
    figure;
    plot(x(:,i),y2,'.'); 
    title(['x_' num2str(i) ' vs. y2']);
    disp('press to continue')
    pause
  end
end
%% independent models
% linear model
LM1 = LinearModel.fit(x,y(:,1));
ypred = LM1.predict(xtest);

disp('prediction for active power')
disp('mae    rmse ')
fprintf('%.4f  %.4f\n', fmae(ytest(:,1),ypred), frmse(ytest(:,1),ypred));
disp('press to continue')
pause

LM2 = LinearModel.fit(x,y(:,2));
ypred = LM2.predict(xtest);
disp('prediction for reactive power')
disp('mae       rmse ')
fprintf('%.4f   %.4f\n', fmae(ytest(:,2),ypred), frmse(ytest(:,2),ypred));

%% fitc model
M = 200;
z0 = x(randperm(size(x,1),M),:);
[z,z0,~,mu1,s2] = learn_fitc(x,y(:,1),xtest,M,z0);
z0 = x(randperm(size(x,1),M),:);
[z,z0,~,mu2,s2] = learn_fitc(x,y(:,2),xtest,M,z0);

disp('mae       rmse ')
fprintf('%.4f   %.4f\n', fmae(ytest(:,1),mu1), frmse(ytest(:,1),mu1));
fprintf('%.4f   %.4f\n', fmae(ytest(:,2),mu2), frmse(ytest(:,2),mu2));



