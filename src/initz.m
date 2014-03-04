function z = initz(x,M,method)
%INITZ  z = initz(x,M,method)
%   Init M inducing inputs given the input x.
[N,D] = size(x);
switch method
  case 'random'
    z = x(randperm(N,M),:);
  case 'espaced'
    if D == 1
      z = linspace(min(x),max(x),M)';
    elseif D == 2
      m = ceil(sqrt(M));  % each dimension has m points
      z1 = linspace(min(x(:,1)),max(x(:,1)),m)';
      z2 = linspace(min(x(:,1)),max(x(:,1)),m)';
      z1 = repmat(z1,1,m);
      z2 = repmat(z2',m,1);
      z = [z1(:),z2(:)];
      z = z(randperm(size(z,1),M),:);
    else
      exit('can only use epsaced for D <= 2');
    end
  case 'kmeans'
    if M <= 100
      [~,z] = kmeans(x,M);
    else
      [~,z] = kmeans(x,100);
      z = [z; x(randperm(N,M-100),:)];
    end
end

end

