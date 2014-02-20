function [A,Knm,Kmminv,Lmm,Kmm] = computeKnmKmminv(covfunc,loghyp,x,z)
%COMPUTEKNMKMMINV [A,Knm,Kmminv,Lmm,Kmm] = computeKnmKmminv(covfunc,loghyp,x,z)
%   Compute the term A = k(x,z)k(z,z)^{-1} which is commonly encountered.
%
Kmm = feval(covfunc, loghyp, z);
Lmm = jit_chol(Kmm,4);
Kmminv = invChol(Lmm);
Knm = feval(covfunc, loghyp, x, z);
A = Knm*Kmminv;
end

