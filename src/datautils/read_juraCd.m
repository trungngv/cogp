function [x,y,xtest,ytest] = read_juraCd()
%READ_JURACD  [x,y,xtest,ytest] = read_juraCd()
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
% output 1,5,7 = Cd,Ni,Zn
y = y(:,[1 5 7]); ytest = ytest(:,[1 5 7]);
y = [y; ytest];
y(N+1:end,1) = nan; % missing 100 inputs of Cd

% pre-process input
[x,xmean,xstd] = standardize(x,[],[]);
xtest = standardize(xtest,xmean,xstd);
end

