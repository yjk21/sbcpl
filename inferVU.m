function [ss, logLik, pred] = inferVU(Y, hyp, junk, options, Ytest) 
% infer z_{in} = [v_i' 1]*[u_n c_n]  + d_i + eps_{in}
% u_n is Gauss(0, covMatU),
% c_n  and d_i are offsets,
% eps is Gauss(0,varianceNoise)
% ss contains Euu = E(v*v') and Eu = E(v) where v is [v_i; d_i]
%
% Written by Emtiyaz, EPFL
% Date modified 

  maxIters = options.maxIters;
  [M2,N] = size(Y);
  M = sqrt(M2);

  % loading matrix
  V = hyp.loadingMat;
  mu = hyp.mean;
  xi = options.warmStart;

  % hyperparameters
  L = hyp.latentDim;
  R = size(V,2) - L;
  Ku = eye(L + R);

  % sufficient stats
  S = zeros(M*L,M*L);
  s = zeros(M*L,1);
  sM = zeros(L+R,1);

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

    % compute loading matrix
    W = B*V(On,:);
    
    % posterior inference
    [Eu, covU, xin] = inferJaakkola(B, W, mu, Ku, Mij, xi(ind,n), maxIters);
    Euu = covU + Eu*Eu';
    xi(ind,n) = xin;

    % collect sufficient stats
    b = -0.5*ones(Mij,1);
    a = (sigmoid(xin) - 0.5)./xin;
    %b = (xin/4 - sigmoid(xin)); 
    %a = ones(Mij,1)/4;
    for m = 1:Mij
      wEuu = a(m)*Euu(1:L,1:L);
      wEu = -b(m)*Eu(1:L); 
      feat = 0;
      if R>0
        feat = a(m)*Euu(1:L,1:R) * W(m,L+(1:R))';
      end
      ii = L*(i(m)-1);
      jj = L*(j(m)-1);
      S(ii+1:ii+L,ii+1:ii+L) = S(ii+1:ii+L,ii+1:ii+L) + wEuu;
      S(jj+1:jj+L,jj+1:jj+L) = S(jj+1:jj+L,jj+1:jj+L) + wEuu;
      S(ii+1:ii+L,jj+1:jj+L) = S(ii+1:ii+L,jj+1:jj+L) + wEuu;
      S(jj+1:jj+L,ii+1:ii+L) = S(jj+1:jj+L,ii+1:ii+L) + wEuu;
      s(ii+1:ii+L) = s(ii+1:ii+L) + wEu - feat; 
      s(jj+1:jj+L) = s(jj+1:jj+L) - wEu - feat; 
    end
    sM = sM + Eu;

    % predition
    miss = find(Ytest(:,n));
    if ~isempty(miss)
      % form B
      nmiss = length(miss);
      [i,j] = ind2sub([M,M], miss);
      Btest = getB(i,j,M);
      Otest = find(sum(Btest~=0,1)~=0); % all items compared (at least once)
      Btest = Btest(:,Otest);
      Wtest = Btest*V(Otest,:);
      % get mean and variance
      diff = Wtest*Eu;%(1:L);
      var = diag(Wtest*covU*Wtest');
      % compute predictive probs
      prob_n = gaussCDFStable(ones(nmiss,1), diff./sqrt(1+var));
      Ytest(miss,n)= prob_n(:);
    end
  end

  % return
  ss.S = S;
  ss.s = s;
  ss.sM = sM;
  ss.warmStart = xi;
  pred.Yhat = Ytest;
  logLik = 0;

return

function [B, On] = getB(i,j,M)
% for i>j, with total M items
  Mij = length(i);
  rows = [1:Mij 1:Mij]';
  cols = [i(:); j(:)];
  vals = [ones(Mij,1); -1*ones(Mij,1)];
  B = sparse(rows, cols, vals, Mij, M);

return

function [m, V, xin] = inferJaakkola(B, W, mu, Ku, Mij, xin, maxIters)

    yn = ones(Mij,1);
    L = size(Ku,1);
    I = eye(L);
    for iter = 1:maxIters
      % find the variational bound params
      b = -0.5*ones(Mij,1);
      a = (sigmoid(xin) - 0.5)./xin;
      %b = (xin/4 - sigmoid(xin)); 
      %a = ones(Mij,1)/4;
      % compute posterior
      U = chol(Ku + W'*diag(a)*W);
      V = U\(U'\I);
      m = V*(W'*(yn + b) + Ku*mu);
      % variational parameters
      mt = W*m;
      vt = diag(W*V*W');
      xin_old = xin;
      xin = sqrt(mt.^2 + vt);
      %xin = mt;
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

