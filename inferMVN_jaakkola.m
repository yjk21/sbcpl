function [ss, logLik, pred] = inferMVN_prefLearn_jaakkola(Y, params, ss, options, Ytest) 

  maxIters = options.maxIters;
  [M2,N] = size(Y);
  M = sqrt(M2);

  % params
  K = params.covMat;
  mu = params.mean;

  % variational params
  xi = options.warmStart;

  % initialize
  S = zeros(M,M);
  s = zeros(M,1);
  logLik = [];
  prob = [];

  % for each n
  for n = 1:N
    % find items such that i > j
    ind = find(Y(:,n)~=0);
    [i,j] = ind2sub([M,M], ind);

    % form B
    B = getB(i,j,M);
    Mij = length(i); 
    On = find(sum(B~=0,1)~=0); % all items compared (at least once)
    B = B(:,On);

    % run variational inference
    xin = xi(ind,n);
    [t, G1, xin] = inferJaakkola(mu(On), K(On,On), B, Mij, xin, maxIters);
%   [t, G1, xin] = inferMeanField(mu(On), K(On,On), B, Mij, [], maxIters);
    xi(ind,n) = xin;

    % compute big data matrix and vector
    S(On,On) = S(On,On) + (t*t' - G1);
    s(On) = s(On) + t;

    % likelihood
    logLik(end+1) = 0;%- 0.5*t'*e - sum(log(diag(U))) - 0.5*Mij*log(2*pi);

    % predition
    miss = find(Ytest(:,n));
    if ~isempty(miss)
      % form B
      nmiss = length(miss);
      [i,j] = ind2sub([M,M], miss);
      Btest = getB(i,j,M);
      Otest = find(sum(Btest~=0,1)~=0); % all items compared (at least once)
      Btest = Btest(:,Otest);
      % get mean and variance
      mean_test = K(Otest,On)*t + mu(Otest);
      var_test = K(Otest,Otest) - K(Otest,On)*G1*K(On,Otest);
      diff = Btest*mean_test;
      var = diag(Btest*var_test*Btest');
      % compute predictive probs
      prob_n = gaussCDFStable(ones(nmiss,1), diff./sqrt(1+var));
      
      Ytest(miss,n) = prob_n(:);
    end
  end
  
 
  % return
  ss.S = S;
  ss.s = s;
  ss.warmStart = xi;
  pred.Yhat = Ytest;



function [t, G1, xin] = inferJaakkola(muo, Koo, B, Mij, xin, maxIters)

    yn = ones(Mij,1);
    for iter = 1:maxIters
      % find the variational bound params
      b = -0.5;
      a = (sigmoid(xin) - 0.5)./xin;
      %b = (xin/4 - sigmoid(xin)); 
      %a = 4*ones(Mij,1);
      % compute G and t
      Kn = B*(Koo*B') + diag(1./a);
      U = chol(Kn);
      Uinv = U\eye(Mij);
      G = Uinv*Uinv';
      e = diag(1./a)*(yn+b) - B*muo;
      t = B'*(G*e);
      G1 = B'*(G*B);
      % marginal mean and variance
      V = Koo - Koo*G1*Koo;
      v = diag(B*V*B');
      m = B*(Koo*t + muo);
      % variational parameters
      xin_old = xin;
      xin = sqrt(m.^2 + v);
      
      
      %xin = m;
      % convergence
      diff = sum(abs((xin_old(:) -xin(:))./xin_old(:)));
      if  diff <1e-4
        break;
      end
      if iter == maxIters
        %fprintf('Maximum iterations reached');
      end
    end

return

function [B, On] = getB(i,j,M)
% for i>j, with total M items
  Mij = length(i);
  rows = [1:Mij 1:Mij]';
  cols = [i(:); j(:)];
  vals = [ones(Mij,1); -1*ones(Mij,1)];
  B = sparse(rows, cols, vals, Mij, M);

return

function [t, G1, xin] = inferMeanField(muo, Koo, B, Mij, xin, maxIters, x0)
  % minfunc options
  optMinFunc.display = 1;
  optMinFunc.maxFunEvals = 100;
  optMinFunc.DerivativeCheck = 'on';
  optMinFunc.Method = 'lbfgs';
  optMinFunc.Tolx = 1e-3;

  bound = getPiecewiseBound('quad', 20);
  bound = rearrangePW(bound);

  Ln = length(muo);
  x0 = [rand(Ln,1); rand(Ln,1)];
  y = ones(Mij,1);
  [x, f] = minFunc(@funObj_meanfield, x0, optMinFunc, y, B, muo, inv(Koo), bound);
  m 

return
