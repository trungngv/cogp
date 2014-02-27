rng(1110, 'twister');
% load demo data set (1D inputs for easy visualization -

%[x,y,xtest,ytest] = load_data('.','toy');
x = load('train_inputs');
y = load('train_outputs');
xtest = load('test_inputs');
M = 10;
[z,z0,~,mu,s2] = learn_fitc(x,y,xtest,M);

plot_all(x,y,xtest,mu,s2,z0,z,'FITC');

