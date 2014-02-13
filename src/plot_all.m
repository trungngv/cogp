function plot_all(x,y,xtest,mu,var,z0,z,titlestr)
%PLOT_ALL plot_all(x,y,xtest,mu,var,z0,z)
%   Plot for analysis.
%
% INPUT
%   - x, y, xtest
%   - mu, var : predictive mean and variance
%   - z0, z : initial and learned inducing
%   - stitle : title string
figure; hold on
plot(xtest,mu,'b') % mean predictions in blue
plotMeanAndStd(xtest,mu,2*sqrt(var),[7 7 7]/7.5);
plot(z0, -0.1 + min(y)*ones(size(z,1), 1), 'k+', 'MarkerSize', 12);
plot(z, -0.2 + min(y)*ones(size(z,1), 1), 'm+', 'MarkerSize', 12);
plot(x,y,'.m') % data points in magenta
ylabel('Predictive distribution');
title(titlestr);

return;
