function [varargout] = inferMVN_ep(varargin)

if nargin == 5
    mode = 1;
end

if mode == 0
    [train, test, params, hypp, epBeta, epPi] = varargin{1:6};
    
    [M,N] = size(train.pos); %number of training pairs, number of users
    [Mt,Nt] = size(test.pos);%number of test pairs, number of users
    
    D = size(params.kmat,1); %number of items
    
    Sigma = hypp.ssq_u * params.vmat *params.vmat' + params.ssq_s * params.kmat;%we could use the MVM method
    
    mu = params.mu;
    
else %support emts code
    [Y, hyp, junk, options, Ytest] = varargin{1:5};

    [Mall, N] = size(Y);
    
    Sigma = hyp.covMat;
    mu = hyp.mean;
    D = length(mu);
    ss.epPi = options.epPi;
    ss.epBeta = options.epBeta;
    ss.warmStart = 0;
    pred.Yhat = Ytest;
    prob = [];
    mpr_n = zeros(N,1);
    %     [train, test, params, hypp, epBeta, epPi]
end


F = zeros(D,D); %allocate stats for upd. Sigma
f = zeros(D,1); %allocate stats for upd. mu
error = 0;

for n = 1:N
    %training pairs
    if mode == 0
        pidx = train.pos(:,n);      nidx = train.neg(:,n);
        %last ep params
        betan = epBeta(:,n);
        pin = epPi(:,n);
        pidxT = test.pos(:,n);      nidxT = test.neg(:,n);
    else
        ind = find(Y(:,n)~=0);
        [pidx,nidx] = ind2sub([D,D], ind);
        
        betan = ss.epBeta(ind, n);
        pin = ss.epPi(ind, n);
        
        M = length(pidx);
        
        indT = find(Ytest(:,n)~=0);
        [pidxT,nidxT] = ind2sub([D,D], indT);
        Mt = length(pidxT);
    end
    
    %input for inference
    BSBt = Sigma(pidx,pidx) + Sigma(nidx,nidx) - Sigma(pidx,nidx) - Sigma(nidx,pidx);
    Bmu = mu(pidx) - mu(nidx);
    
    
    %run EP approximate inference
    [betan, pin, E, e] = runEP(full(betan), full(pin), BSBt, Bmu);
    
    %set new ep params
    if mode == 0
        epBeta(:,n) = betan;
        epPi(:,n) = pin;
    else
        ss.epBeta(ind,n) = betan;
        ss.epPi(ind,n) = pin;
    end
    %accumulate F, f: B'*w doesnt seem as simple as B*v, TODO is it??
    Bn = sparse([1:M,1:M]', [pidx;nidx], [ones(M,1);-ones(M,1)], M, D);
    
    %accumulation is
    %   - f += Bn' * e
    %   - F += Bn' * (E'*E - e*e') * Bn = Bn'*E'*E*Bn - Bn'*e*e'*Bn

    ftemp = Bn' * e;         Ftemp = Bn' * E';

    
    F = F + Ftemp * Ftemp'- ftemp  * ftemp';
    f = f + ftemp;
    
    %calculate log probs and predictions
    
    if ~isempty(pidxT)
      BsS = Sigma(pidxT,:) - Sigma(nidxT,:); %Bs * Sigma
      Bsmu = mu(pidxT) - mu(nidxT);          %Bs * mu
      predMu = Bsmu + BsS * ftemp; %predictive mean E_Q[x_test]
      %for predictive dist, we need Var_Q[x_test] as well
      tempPredS = BsS  * Ftemp;
      BsSBst = Sigma(pidxT,pidxT) + Sigma(nidxT,nidxT) - Sigma(pidxT,nidxT) - Sigma(nidxT,pidxT);
      predVars = diag(BsSBst) - sum(tempPredS .* tempPredS,2); %pred. variances
      yTest = []; %if empty, it is assumed that ytest = 1
      zTest = predMu ./ sqrt(1 + predVars);
      pPred = GaussCDFStable(yTest, zTest);%predictive probs P(ytest | y)
      
      % compute MPR
      pos = unique(pidxT);
      lenIh = length(pos);
      for ii = 1:lenIh
        idx = find(pidxT==pos(ii));
        lenJh = length(idx);
        mpr_n(n) = mpr_n(n) + sum(pPred(idx))/lenJh;
      end
      mpr_n(n) = mpr_n(n)/lenIh;

      if mode ~= 0
          pred.Yhat(indT,n) = pPred;
          prob = [prob; pPred(:)];
      end
      error = error +  sum(predMu <= 0);
      if mod(n, 1000) == 0
          %fprintf('%04d: %f, %d; ',n,error / (Mt * N), error)
      end
    end
end
%fprintf('\n')

%final statistics according to spec


if mode ==0
    f = f ./ N;
    F = F./ N + f * f';
    
    error = error / (Mt * Nt);
    varargout = {F,f,error, epBeta, epPi};
else
    ss.S = -F;
    ss.s = f;
    pred.mpr = mpr_n;
    varargout = {ss,0,pred};
end

end

%% EP Inference
function [betan, pin, E, e] = runEP(betan, pin, BSBt, Bmu)

nEP = 7;

M = length(pin);%number of potentials (Ln)

I = speye(M);
y = ones(M,1);
K = BSBt;
m = Bmu;
ttau = pin;
tnu = betan;
for it = 1:nEP
    % - calculate posterior moments
    %     [Sigma, mu, nlZ, L] = epComputeParams(K, y, ttau, tnu, m);
    sqrtTtau = sqrt(ttau);
    L = chol(I + (sqrtTtau * sqrtTtau') .* K,'lower') ;
    
    V = L \ (diag(sqrtTtau)*K);
    %     s = 1./(diag(K) - sum(V.*V,2));
    Sigma = K - V'*V;
    s = 1./diag(Sigma);
    
    E = L \ diag(sqrtTtau);
    mu = m + K *(tnu - E'*E*(K*tnu  + m));
    
    
    % - calculate cavity parameters
    %     s = 1./diag(Sigma);
    tau_ni = s - ttau;  %  first find the cavity distribution ..
    %     nu_ni = mu .* s + m .* tau_ni - tnu;% .. params tau_ni and nu_ni
    nu_ni = mu .* s - tnu;% .. params tau_ni and nu_ni
    
    % - calculate new site parameters
    [lZ, dlZ, d2lZ] = likErf(y, nu_ni./tau_ni, 1./tau_ni);
    
    ttau = max( -d2lZ  ./ (1 + d2lZ./tau_ni), 1e-12); % enforce positivity i.e. lower bound ttau by zero
    %     tnu  = ( dlZ + (m-nu_ni./tau_ni).*d2lZ)./(1 + d2lZ./tau_ni);
    tnu  = ( dlZ + (-nu_ni./tau_ni).*d2lZ)./(1 + d2lZ./tau_ni);
end
varXinv = s;
meanX = mu;
betan = tnu;
pin = ttau;
sqrtPin = sqrt(pin);
bbp = betan ./ (sqrtPin+0.001);

% fprintf('MM:(v:%.016f; m:%.016f)\n',norm((varXinv - (pin+tau_ni))./max(max(varXinv, (pin+tau_ni)),1e-12),'inf'), norm((meanX - (dlZ+nu_ni) ./tau_ni  ),'inf'))

%  fprintf('\n')
%stats in terms of z
L = chol(I + diag(sqrtPin) * BSBt * diag(sqrtPin), 'lower');
E = L \ diag(sqrtPin);
% e = E' * (L \ (bbp - Bmu .* sqrtPin));
e = betan - E' * E * (BSBt * betan + Bmu);

end

%% Probit Likelihood Potential: services for EP update
function [lZ,dlZ,d2lZ] = likErf(y, mu, s2)
z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
[junk,lZ] = GaussCDFStable(y,z);                             % log part function
if numel(y)>0, z=z.*y; end
if nargout>1
    if numel(y)==0, y=1; end
    n_p = gaussOverGaussCDF(z,exp(lZ));
    dlZ = y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
    if nargout>2
        d2lZ = -n_p.*(z+n_p)./(1+s2);                % 2nd derivative wrt mean
    end
end

end

%% Util: Gauss CDF, implemented using error function
function [p,lp] = GaussCDFStable(y,f)
if numel(y)>0, yf = y.*f; else yf = f; end     % product of latents and labels
p  = (1+erf(yf/sqrt(2)))/2;                                       % likelihood
if nargout>1, lp = logphi(yf,p); end                          % log likelihood
end

%% Util: safe implementation of the log of phi(x) = \int_{-\infty}^x N(f|0,1) df logphi(z) = log(normcdf(z))
function lp = logphi(z,p)
lp = zeros(size(z));                                         % allocate memory
zmin = -6.2; zmax = -5.5;
ok = z>zmax;                                % safe evaluation for large values
bd = z<zmin;                                                 % use asymptotics
ip = ~ok & ~bd;                             % interpolate between both of them
lam = 1./(1+exp( 25*(1/2-(z(ip)-zmin)/(zmax-zmin)) ));       % interp. weights
lp( ok) = log( p(ok) );
% use lower and upper bound acoording to Abramowitz&Stegun 7.1.13 for z<0
% lower -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+2   ) -z/sqrt(2) )
% upper -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+4/pi) -z/sqrt(2) )
% the lower bound captures the asymptotics
lp(~ok) = -log(pi)/2 -z(~ok).^2/2 -log( sqrt(z(~ok).^2/2+2)-z(~ok)/sqrt(2) );
lp( ip) = (1-lam).*lp(ip) + lam.*log( p(ip) );
end

%% Util: Stable computation of N(x) / Phi(x)
function n_p = gaussOverGaussCDF(f,p)
n_p = zeros(size(f));       % safely compute Gaussian over cumulative Gaussian
ok = f>-5;                            % naive evaluation for large values of f
n_p(ok) = (exp(-f(ok).^2/2)/sqrt(2*pi)) ./ p(ok);

bd = f<-6;                                      % tight upper bound evaluation
n_p(bd) = sqrt(f(bd).^2/4+1)-f(bd)/2;

interp = ~ok & ~bd;                % linearly interpolate between both of them
tmp = f(interp);
lam = -5-f(interp);
n_p(interp) = (1-lam).*(exp(-tmp.^2/2)/sqrt(2*pi))./p(interp) + lam .*(sqrt(tmp.^2/4+1)-tmp/2);

end
