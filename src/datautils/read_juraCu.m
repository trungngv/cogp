function [x,y,xtest,ytest] = read_juraCu()
%READ_JURACU  [x,y,xtest,ytest] = read_juraCu()
%   
clear; load jura_all
% original jura dataset
% 4D inputs: xloc,yloc,landuse,rock
% 7 outputs: Cd,Co,Cr,Cu,Ni,Pb,Zn

% construct similar dataset as in gprn paper
N = size(x,1);
x = x(:,[1 2]);   % use xloc & yloc only
xtest = xtest(:,[1 2]);
x = [x; xtest];
% outputs = Cu,Ni,Pb,Zn
y = y(:,[4:7]); ytest = ytest(:,[4:7]);
y = [y; ytest];
y(N+1:end,1) = nan; % missing 100 inputs of Cd

% pre-process input
[x,xmean,xstd] = standardize(x,[],[]);
xtest = standardize(xtest,xmean,xstd);
end

