% Given the parameters of the convolved GP model (LQR, L), compute the
% parameters Kf, C for multi-task GP that approximates the convolved
% processes with Laplace approximation.

% Trung Nguyen
% May 2012
function [Kf C Cavg] = laplace_approx_cgp(S, LQR, L)
  Q = size(LQR, 1);
  D = size(LQR{1, 1}, 1);
  R = length(L);
  C = cell(Q, Q);
  Cavg = zeros(D, D);
  Kf = zeros(Q);
  SSL = zeros(R);
  LLL = cell(R, 1);
  % Pre-compute inv(L{r}) for re-usability
  Linv = cell(R, 1);
  for r = 1:R
    Linv{r} = pinv(L{r});
  end
  for q = 1:Q
    for s = 1:Q
      % Compute SSL(r) and LLL{r} for this pair of q and s
      % SSL(r) = S_{qr}S_{sr}det(inv(L{r}))^0.5
      % LLL{r} = inv(LQR{q}) + inv(LQR{s}) + inv(L{r});
      partition = 0;
      C{q, s} = zeros(D, D);
      for r = 1:R
        SSL(r) = S(q,r) * S(s,r) * sqrt(det(Linv{r}));
        Kf(q,s) = Kf(q,s) + SSL(r);
        LLL{r} = pinv(LQR{q,r}) + pinv(LQR{s,r}) + Linv{r};
        weight = SSL(r) / sqrt(det(LLL{r}));
        C{q, s} = C{q, s} + weight * pinv(LLL{r});
        partition = partition + weight;
      end
      C{q, s} = C{q, s} / partition;
      Cavg = Cavg + C{q, s};
    end
  end
  % Approximate C from all C for all pair of tasks by taking their mean
  Cavg = Cavg / (Q * Q);
  
end
