function [i,p,L,Q,R,d,dnlZ] = cholqr_optim(i,hyp,cov,mean,x,y,opt)

% TODO: approximate scoring to speed up computations
%
% The code is based on the paper: Efficient optimization for sparse Gaussian
% process regression, NIPS 2013 by Y. Cao, M. Brubaker, H. Hertzmann & D. Fleet
%
% Copyright (c) by Yanshuai Cao and Hannes Nickisch 2013-11-16.

l = opt.ncand;                                         % optimisation parameters
J = opt.nswap;
q = opt.nskip;
o = opt.out;

sn = exp(hyp.lik); snu = sn/1e3;  % signal and inducing point standard deviation
gppar = {cov,hyp.cov,x,snu,sn};  % compact form GP covariance related parameters
[p,L,Q,R,d] = cholqr(i,gppar{:});               % compute initial representation
n = numel(y); m = numel(i);                                         % dimensions
neq = 0;
dnlZ = 0;
y = y - feval(mean{:},hyp.mean,x);          % allow for arbitrary mean functions
for j=1:J
  k = randperm(m); k = k(1); pk = p(k);                 % sample point to remove
  [p,L,Q,R,d] = cholqr_remove(p,L,Q,R,d, k);
  cc = true(n,1); cc(p(1:m)) = false; cc = find(cc);         % available indices
  c = randperm(n-m); c = c(1:l);                             % sample candidates
  dnlZj = cholqr_dnlZ([cc(c);pk],L,Q,gppar{:},y);            % candidate scoring
  dnlZj = dnlZj-dnlZj(l+1);          % subtract effect of adding removed point k
  [dnlZj,ci] = min(dnlZj);                       % obtain best scoring candidate
  if ci==l+1, neq = neq+1; pnew = pk;                % no better candidate found
  else        neq = 0;     pnew = cc(c(ci)); end        % better candidate found
  [p,L,Q,R,d] = cholqr_insert(p,L,Q,R,d,pnew,gppar{:});       % insert new point
  if abs(dnlZj)<1e-9, dnlZj = 0; else dnlZj = -dnlZj; end
  if o, fprintf('%03d) dlZ=%1.3e %d->%d\n',j,dnlZj,pk,pnew), end % report progr.
  if (neq>q) && o, fprintf('break\n'), break, end
  dnlZ = dnlZ + dnlZj;                                     % accumulate progress
end
i = p(1:m);

function [p,lk] = cholqr_swap_lk(j,k,p,cov,hyp,x,snu)
% swap indices in p and compute Cholesky column lk, O(n)
  t = find(p==j); pk = p(t); p(t) = p(k); p(k) = pk;          % swap the indices
  lk = feval(cov{:},hyp,x,x(j,:)); lk(j) = lk(j) + snu^2;   % compute covariance

function [L,Q,R,d] = cholqr_update(p,L,Q,R,d, k, sn)
% augment L if required, modify k-th columns of Q,R and update d, O(n*m)
  n = numel(p);                                                     % dimensions
  d(p(k:n)) = d(p(k:n)) - L(p(k:n),k).^2;                      % update diagonal
  if nargin>6, L(n+k,k) = sn; end                              % do augmentation
  R(1:k-1,k) = Q(:,1:k-1)'*L(:,k);                                  % do QR part
  Q(:,k) = L(:,k) - Q(:,1:k-1)*R(1:k-1,k);
  R(k,k) = norm(Q(:,k));
  Q(:,k) = Q(:,k)/R(k,k);

function [p,L,Q,R,d] = cholqr(i,cov,hyp,x,snu,varargin)
% compute representation from scratch so that post conditions hold, O(n*m^2)
  m = numel(i); n = size(x,1);                                      % dimensions
  if nargin>5, L = zeros(n+m,m); Q = zeros(n+m,m);  % augmentation if sn present
  else         L = zeros(n  ,m); Q = zeros(n  ,m);  end
  p = 1:n; R = zeros(m,m);                                                % init
  d = feval(cov{:},hyp,x,'diag') + snu^2;      % diagonal of the full covariance
  for k=1:m
    [p,lk] = cholqr_swap_lk(i(k),k,p,cov,hyp,x,snu); % swap indices p and get lk
    lk = lk(p(k+1:n)) - L(p(k+1:n),1:k-1)*L(p(k),1:k-1)';
    L(p(k:n),k) = [d(p(k)); lk]/sqrt(d(p(k)));                 % update Cholesky
    [L,Q,R,d] = cholqr_update(p,L,Q,R,d, k, varargin{:});   % augment, update QR
  end

function [p,L,Q,R,d] = cholqr_remove(p,L,Q,R,d, k)
% remove k-th inducing point by moving it to the rightmost column of L,Q,R
% and setting this column to zero; post conditions do not hold, O(n*m^2)
  if k<size(R,1), [p,L,Q,R,d] = cholqr_permuteright(p,L,Q,R,d, k); end
  [L,Q,R,d] = cholqr_removelast(p,L,Q,R,d);

function [Q,R] = qr2(M)
% QR decomposition for 2x2 matrices
  Q = zeros(2,2);
  X = sqrt( M(1,1)^2 + M(2,1)^2 );
  R = X;
  Q(:,1) = M(:,1) / X;
  R(1,2) = Q(:,1).' * M(:,2);
  Q(:,2) = M(:,2) - R(1,2) * Q(:,1);
  R(2,2) = norm(Q(:,2));
  Q(:,2) = Q(:,2) / R(2,2);

function [p,L,Q,R,d] = cholqr_permuteright(p,L,Q,R,d, k)
% if k<m pivot column k to rightmost column of L,Q,R; post conditions still hold
  m = size(R,1); n = numel(p);                                      % dimensions
  for s=k:m-1
    ps = p(s); p(s) = p(s+1); p(s+1) = ps;                        % swap indices
    [Q1,R1] = qr2(L(p(s:s+1),s:s+1)');
    L(p(s:n),s:s+1) = L(p(s:n),s:s+1)*Q1;
    L(p(s),s+1) = 0;
    R(1:m,s:s+1) = R(1:m,s:s+1)*Q1;
    [Q2,R2] = qr2(R(s:s+1,s:s+1));
    R(s:s+1,1:m) = Q2'*R(s:s+1,1:m);
    Q(:,s:s+1) = Q(:,s:s+1)*Q2;
    R(s+1,s) = 0;
    if size(L,1)>n, Q(n+(s:s+1),1:m) = Q1'*Q(n+(s:s+1),1:m); end       % augment
  end

function [L,Q,R,d] = cholqr_removelast(p,L,Q,R,d)
% remove last (rightmost) column from L,Q,R
  m = size(R,1); n = numel(p);
  d(p(m:n)) = d(p(m:n)) + L(p(m:n),m).^2;
  L(:,m) = 0; Q(:,m) = 0; R(1:m,m) = 0; R(m,1:m) = 0;

function dnlZj = cholqr_dnlZ(j, Lm,Qm, cov,hyp,x,snu,sn,y)
% compute change in marginal likelihood adding points j
% Lm and Qm have both size (m+n,m) but their last column is supposed to be zero
  n = numel(y); m = size(Qm,2);                                     % dimensions
  yt = y; yt(m+n) = 0;                                               % augment y
  dnlZj = zeros(size(j));                                      % allocate memory
  for ii=1:numel(j)
    i = j(ii);
    lm = feval(cov{:},hyp,x,x(i,:));
    lm(i) = lm(i) + snu^2;
    lm = lm - Lm(1:n,:)*Lm(i,:)'; lm = lm/sqrt(lm(i)); 
    lmt = lm; lmt(m+n) = sn;
    v = lmt - Qm*(Qm'*lmt); qm = v/sqrt(v'*v);
    dnlZj(ii) = log(v'*v)/2 - log(sn) - ((yt'*qm)^2+lm'*lm)/(2*sn^2);
  end

function [p,L,Q,R,d] = cholqr_insert(p,L,Q,R,d, j, cov,hyp,x,snu,varargin)
% insert inducing point j into empty rightmost column of L,Q,R;
% post conditions hold, O(n*m^2)
  m = size(R,1); n = numel(p);
  [p,lk] = cholqr_swap_lk(j,m,p, cov,hyp,x,snu);     % swap indices p and get lk
  lk = lk - L(1:n,:)*L(j,:)';
  L(1:n,m) = lk/sqrt(lk(j));                                   % update Cholesky
  [L,Q,R,d] = cholqr_update(p,L,Q,R,d, m, varargin{:});     % augment, update QR
