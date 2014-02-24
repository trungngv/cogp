function chkgrad_test()
%CHKGRADELBO chkgrad_svi_elbo()
%  Check gradient of the evidence lowerbound wrt covariance hyperparameters
%  and inducing inputs.
%
% See also
%   svi_elbo

D = 4;
theta = rand(D,1);
[g, delta] = gradchek(theta', @f, @grad);
disp(delta)
end

function fval = f(x)
%covPeriodic = sf2 * exp( -2*sin^2( pi*||x-y||/p )/ell^2 )
x = x';
sf2 = 0.5;
p = 1.5;
ell2 = 0.3;
y = [1;2;3;4];
hyp = [log(sqrt(ell2)); log(p); log(sqrt(sf2))];
sin2 = sin(pi*sqrt((x-y)'*(x-y))/p)^2;
fval = sf2*exp(-2*sin2/ell2);
fval2 = covPeriodic(hyp,x',y');
disp(['diff = ' num2str(abs(fval-fval2))])
end

function g = grad(x)
x = x';
sf2 = 0.5;
p = 1.5;
ell2 = 0.3;
y = [1;2;3;4];
r = pi*sqrt(sq_dist(x,y))/p;
for i=1:numel(x)
  g(i) = -f(x')*sin(r)*cos(r)*(x(i)-y(i))/sqrt(sq_dist(x,y))*(4*pi/p/ell2);
end
end
