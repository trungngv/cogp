function [x,y,xtest,ytest,y0] = read_fx()
%READ_FX  [x,y,xtest,ytest,y0] = read_fx()
%   Read the exchange rate dataset as described in Alvarez.
% 
%   No pre-processing of inputs and outputs is performed here.
clear all;
y = csvread('data/fx/fx2007-processed.csv');
y = 1./y; % usd / currency
y0 = y;
x = (1:size(y,1))';
y(y == -1) = nan; % missing data

xtest = x;
ytest = y(xtest,:);

% imput missing data
% CAD = 4, JPY = 6, AUD = 9
y(50:100,4) = nan;
y(100:150,6) = nan;
y(150:200,9) = nan;
