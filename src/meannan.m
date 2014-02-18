function m = meannan(x)
%MEANNAN  m = meannan(x)
%   Replacement for the mean() function where nan is regarded as the mean
%   of all valid elements (i.e. not nan) in the same column.
%

% column wise
for i=1:size(x,2)
  valid = ~isnan(x(:,i));
  x(isnan(x(:,i)),i) = mean(x(valid,i));
end
m = mean(x,2);
end

