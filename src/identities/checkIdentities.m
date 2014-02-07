% Check some identities
%

%% (ABA')_n,n = a(n,:)*B*a(n,:)'
A = rand(200,100);
B = rand(100,100);
ABA = A*B*A';
totaldiff = 0;
for n=1:size(A,1)
  assert(abs(ABA(n,n) - A(n,:)*B*A(n,:)') < 1e-10, ['diff = ' num2str(abs(ABA(n,n) - A(n,:)*B*A(n,:)'))]);
  totaldiff = totaldiff + abs(ABA(n,n) - A(n,:)*B*A(n,:)');
end
disp('total diff')
disp(totaldiff)

%% AB'BA = \sum_{n=1}^N AB(n,:)'B(n,:)A
% A : M x M (like k_MM)
% B : N x M (like k_NM)
A = rand(1000,1000);
B = rand(2000,1000);
ABBA = A*(B')*B*A;
result = zeros(size(A));
for n=1:size(B,1)
  result = result + A*(B(n,:)')*B(n,:)*A;
end
disp('total diff')
disp(norm(ABBA - result, 2))

