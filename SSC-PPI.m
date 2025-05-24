function [X_new, W, H, iter, obj, idx, RW, RH] = newFS(X, c, alpha, beta, l, maxIter)
[m,n]=size(X);
H = orth(rand(n,c));
W = orth(rand(m,c));
A=X'*X;
for iter=1:maxIter
HH = H*H';
E = X'*W - H;
F = A - HH;
Ei = sqrt(sum(E.*E,2) + eps);
p = 0.5./Ei;
P = diag(p);
Fi = sqrt(sum(F.*F,2) + eps);
q = 0.5./Fi;
Q = diag(q);
W = W.*((2*X*P*H)./(2*X*P*X'*W + (beta./(1 + W)) + eps));
H = H.*((P*X'*W + alpha*A*Q*H + alpha*Q*A*H)./(P*H + 2*alpha*HH*Q*H));
W = max(W, 0); H = max(H, 0);
[W, H]=t_rank(W, H, c);
obj(iter) = sum(Ei) + alpha*sum(Fi) + trace((A-HH)'*(A-HH)) + beta*sum(sum(log(1 + W)));
% if iter > 2
% if abs(obj(iter) - obj(iter - 1))/obj(iter - 1) < 1e-30
% break
% end
% end
end
RW = rank(W); RH = rank(H);
score = sqrt(sum(W.*W,2));
[~, idx] = sort(score,'descend');
X_new = X (idx(1:l),:);
X_new = X_new';
function [W, H] = t_rank(W, H, C)
rw = rank(W);
if rw ~= C
% disp('The matrix W is not full rank, so you can get an approximate optimal result.')
[W, ~, VU] = svds(W, C);
H = H*VU';
end
rh = rank(H);
if rh ~= C
% disp('The matrix H is not full rank, so you can get an approximate optimal result.')
[H, ~, VH] = svds(H, C);
W = W*VH';
end
end
end