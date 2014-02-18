function s = sumbeta(params)
%SUMBETA  s = sumbeta(params)
%   s = \sum params.task{i}.beta
P = size(params.task,1);
s = 0;
for i=1:P
  s = s + params.task{i}.beta;  
end

end

