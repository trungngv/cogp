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
plotMeanAndStd(xtest,mu,2*sqrt(var),[7 7 7]/9);
if ~isempty(z0)
  plot(z0, -0.5 + min(y)*ones(size(z0,1), 1), 'k+', 'MarkerSize', 12);
end
if ~isempty(z)
  plot(z, -0.5 + min(y)*ones(size(z,1), 1), 'm+', 'MarkerSize', 12);
end  
plot(x,y,'.m','MarkerSize',14) % data points in magenta
%ylabel('Predictive distribution');
title(titlestr,'FontSize',20);
set(gca, 'FontSize', 20);

return;
