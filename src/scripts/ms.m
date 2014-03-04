rng(1110,'twister');
[x,y,xtest,ytest] = load_data('data/ms','ms');

[x,xmean,xstd] = standardize(x,[],[]);
xtest = standardize(xtest,xmean,xstd);

fmae = @(ytrue,ypred) mean(abs(ypred-ytrue(:,1)));
frmse = @(ytrue,ypred) sqrt(mean((ypred-ytrue(:,1)).^2));

% linear model
LM = LinearModel.fit(x,y(:,1));
ypred = LM.predict(xtest);

% gp model
GM = standard_gp([],x,y(:,1),xtest,[],false);
disp('lengthscales')
disp(exp(GM.hyp.cov));

disp('prediction for Pulse1')
disp('mae    rmse ')
fprintf('%.4f  %.4f\n', fmae(ytest(:,1),ypred), frmse(ytest(:,1),ypred));
fprintf('%.4f   %.4f\n', fmae(ytest(:,1),GM.fmean), frmse(ytest(:,1),GM.fmean));
disp('press to continue')
pause

% pulse 2
LM = LinearModel.fit(x,y(:,2));
disp('prediction for Pulse2')
ypred = LM.predict(xtest);

GM = standard_gp([],x,y(:,2),xtest,[],false);
disp('lengthscales')
disp(exp(GM.hyp.cov));
disp('prediction for Pulse2')
disp('mae       rmse ')
fprintf('%.4f   %.4f\n', fmae(ytest(:,2),ypred), frmse(ytest(:,2),ypred));
fprintf('%.4f   %.4f\n', fmae(ytest(:,2),GM.fmean), frmse(ytest(:,2),GM.fmean));


