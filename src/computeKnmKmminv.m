function [A,Knm,Kmminv,Lmm,Kmm] = computeKnmKmminv(covfunc,loghyp,x,z)
%COMPUTEKNMKMMINV [A,Knm,Kmminv,Lmm,Kmm] = computeKnmKmminv(covfunc,loghyp,x,z)
%   Compute the term A = k(x,z)k(z,z)^{-1} which is commonly encountered.
%
Kmm = feval(covfunc, loghyp, z) + 1e-10*eye(size(z,1));
Lmm = jit_chol(Kmm);
Kmminv = invChol(Lmm);
Knm = feval(covfunc, loghyp, x, z);
A = Knm*Kmminv;
end

