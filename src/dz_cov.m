function [DKmm,DKmn] = dz_cov(covfunc,loghyp,x,z,i,Kmm,Knm)
%DZ_COV dz = dz_cov(covfunc,loghyp,x,z,i)
%   
% Derivatives of the covariance function wrt inducing inputs.
% Warning: this code is vectorized for internal use of svi only.
switch covfunc
  case 'covSEard'
    ell2 = exp(2*loghyp(1:end-1));
    DKmm = bsxfun(@minus,z(:,i),z(:,i)');
    DKmm = -(Kmm.*DKmm)/ell2(i);
    DKmn = bsxfun(@minus,z(:,i),x(:,i)');
    DKmn = -(Knm'.*DKmn)/ell2(i);
  case 'covPeriodic'
    ell2 = exp(2*loghyp(1));
    pval = exp(loghyp(2));
    DKmm = bsxfun(@minus,z(:,i),z(:,i)');
    normz = sqrt(sq_dist(z')) + 1e-15*eye(size(z,1));
    r = pi*normz/pval;
    DKmm = -(Kmm.*sin(r).*cos(r).*DKmm./normz)*(4*pi/pval/ell2);
    DKmn = bsxfun(@minus,z(:,i),x(:,i)');
    normzx = sqrt(sq_dist(z',x'));
    r = pi*normzx/pval;
    DKmn = -(Knm'.*sin(r).*cos(r).*DKmn./normzx)*(4*pi/pval/ell2);
  case 'covLINard'
    ell2 = exp(2*loghyp);
    DKmm = repmat(z(:,i)',size(z,1),1)/ell2(i);
    DKmm = setDiag(DKmm,2*diag(DKmm));
    DKmn = repmat(x(:,i)',size(z,1),1)/ell2(i);
  otherwise
    error('covariance function not supported');
end

end

