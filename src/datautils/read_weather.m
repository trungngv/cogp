function [x,y,xtest,ytest,y0] = read_weather()
%READ_WEATHER  [x,y,xtest,ytest] = read_weather()
%   Reads the weather data.
clear all;
output = 4; % air temperature
load bra
ybra = y(:,output);
load cam
ycam = y(:,output);
load chi
ychi = y(:,output);
load sot
ysot = y(:,output);

range = x >= 10 & x <= 15;
x = x(range);
y = [ybra,ycam,ychi,ysot];
y = y(range,:);

y(y == -1) = nan; % missing data
y0 = y;

xtest = x;
ytest = y;

% impute missing data
y(x >= 10.2 & x <= 10.8,2) = nan;
y(x >= 13.5 & x <= 14.2,3) = nan;

