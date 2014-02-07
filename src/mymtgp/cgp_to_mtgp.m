% Given the parameters of the convolved GP model (LQR, L), compute the
% parameters Kf, C for multi-task GP that approximates the convolved
% processes. Here we assume only one latent function.
%
% Deprecated.
% This is only a special case of the approximateConvolvedGP.m function

function [Kf C Cavg] = convolvedGPToMultiTaskGP(S, LQR, L)
  Linv = pinv(L{1});
  Q = size(LQR, 1);
  D = size(LQR{1, 1}, 1);
  C = cell(Q, Q);  
  Cavg = zeros(D, D);
  Kf = zeros(Q);
  for q = 1:Q
    for s = 1:Q
      C{q, s} = pinv(LQR{q,1}) + pinv(LQR{s,1}) + Linv;
      Kf(q, s) = S(q,1) * S(s,1) * (det(Linv)^0.5);
      Cavg = Cavg + C{q, s};
    end
  end
  % Approximate C from all C for all pair of tasks by taking their mean
  Cavg = Cavg / (Q * Q);
end
