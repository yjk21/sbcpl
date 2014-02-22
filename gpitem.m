function [logLoss] = gpitem(method, dataName, nLatentDims, useFeatures, gpScale, gpSigma, seed, Ytrain, Ytest, Xm, Xu)

  % hyperparameters
  s2 = exp(gpSigma); % sigma_s^2
  hyp.cov = [gpScale gpSigma]; % GP kernel params
  kappa = 0;%100;
  zitter = 1e-1; % epsilon
  workerPath = '~/Dropbox/matrixFactorization/ipcFiles/ipcWorker';
  outDir = './resultsAIstats/';
  fileName = sprintf('%s/%s_%s_%d_%d_%d_%d', outDir, method, dataName, nLatentDims, useFeatures, gpScale, seed);
  saveOut = 0;

  % get data
  setSeed(seed);
%   dataDirName = '/lapmal/project/datasets/matrixFact/';
%   [Ytrain, Ytest, Xm, Xu] = getDataPrefLearn(dataName, dataDirName, []);
  features = struct('Xm',Xm,'Xu',Xu);
  miss = find(Ytest~=0);
  nMiss = length(miss);
  obs = find(Ytrain~=0);
  [M2,N] = size(Ytest);
  M = sqrt(M2);

  % inference method
  jaakkola = 0;
  ep = 0; 
    % it is the PPCA model
    % now initialize model params
    mu = zeros(M,1);
    VV = eye(M);
    Kz = zitter*eye(M); % kappa is added for K to posdef when no features
    Km = kappa*(ones(M,M));
    if useFeatures
        Km = Km + covSEiso(hyp.cov, Xm);
        Km = 0.5*(Km + Km');
    end
    %eig(Km+Kz)
    %imagesc(Km+Kz); colorbar;
    cholKm = chol(Km + Kz,'lower');
    %cond(Km+Kz)
    %cond(cholKm)
    %pause
    hyp.mean = mu;
    hyp.covMat = 0*VV + s2*(Km + Kz);

    % specify infer functions
    switch method
    case {'ppcaJak','npcaJak'}
      inferFun = @inferMVN_jaakkola;
      jaakkola = 1;
    case {'npcaEp','ppcaEp'}
      inferFun = @inferMVN_ep;
      ep = 1;
   otherwise
      error('Unknown inference method')
    end

  % initialize variational parameters for binary data
  setSeed(999);
  obs = find(Ytrain~=0);
  % for jaakkola bound
  if jaakkola
      xi = Ytrain;
      xi(obs) = rand(length(obs(:)),1);
      options.warmStart = xi;
      options.maxIters = 100;
  end
  % for EP
  if ep
      options.epBeta = Ytrain;
      options.epPi = Ytrain;
      options.epBeta(obs) = 1e-10;
      options.epPi(obs) = 1;
      options.maxIters = 100;
      options.workerPath = workerPath;
  end

  % run EM
  %fprintf('\nRunning %s\n',method);
  %fprintf('iter AUC logLoss\n');
  iter = 1;

      % infer preferences
      [ss, logLik, pred] = inferFun(Ytrain, hyp, features, options, Ytest);
     
      % prediction
      prob = full(pred.Yhat(miss));
      nMiss = length(prob);
      ii = floor(nMiss/2);
      [X1,X2,T,auc(iter)] = perfcurve([ones(ii,1); -ones(nMiss-ii,1);], [prob(1:ii);1-prob(ii+1:end)], 1);
      logLoss(iter) = mean(-log2(prob));
      %fprintf('%d %0.4f %.4f (s2: %.3f)\n', iter, auc(iter), logLoss(iter), s2);


