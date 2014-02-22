function [logLoss, s2, tt] = prefLearnMethods(method, dataName, nLatentDims, useFeatures, gpScale, seed, Ytrain, Ytest, Xm, Xu)
    %[logLoss, s2, tt] = exptPref_comp(method, dataName, nLatentDims, useFeatures, gpScale, seed, Ytrain, Ytest, Xm, Xu)
    %
    % DESCRIPTION: implementation of methods for AISTATS 2014 
    %   
    % INPUT
    % - method
    % - dataName
    % - nLatentDims
    % - useFeatures
    % - gpScale
    % - seed
    % - Ytrain
    % - Ytest
    % - Xm
    % - Xu
    %
    % OUTPUT
    % - logLoss
    % - s2
    % - tt



    fprintf('D %d UseFeat %d Seed %d\n', nLatentDims, useFeatures, seed);
    maxIters = 100;

    % hyperparameters
    s2 = 0; % sigma_s^2
    hyp.cov = [gpScale 0]; % GP kernel params
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
    switch method
        case 'VU'
            % v*u model
            % now initialize model params
            setSeed(555);
            L = nLatentDims;
            V = rand(M,L);
            hyp.loadingMat = V;
            hyp.mean = zeros(L,1);
            hyp.latentDim = nLatentDims;
            % specify inferFun options
            jaakkola = 1;
            inferFun = @inferU_prefLearn_1;

        case 'S' 
            D_x = size(Xm,2);        % number of item features
            D_t = size(Xu,2);        % number of user features
            hyp.theta = [zeros(D_x +1 + D_t + 1,1); -2]; % same settings used by Bonilla
            options.userCovfunc = 'covSEard_bonilla';
            options.itemCovfunc = 'covSEard_bonilla';
            % only GP model
            inferFun = @runGPPE;
            maxIters = 1;

        otherwise
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
            hyp.covMat = VV + s2*(Km + Kz);

            % specify infer functions
            switch method
                case {'ppcaJak','npcaJak'}
                    inferFun = @inferMVN_prefLearn_jaakkola;
                    jaakkola = 1;
                case {'npcaEp','ppcaEp'}
                    inferFun = @inferMVN_ep;
                    ep = 1;
                otherwise
                    error('Unknown inference method')
            end
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
    fprintf('\nRunning %s\n',method);
    fprintf('iter AUC logLoss\n');
    for iter = 1:maxIters
        tic;
        % infer preferences
        [ss, logLik, pred] = inferFun(Ytrain, hyp, features, options, Ytest);

        % update warm start params
        if jaakkola
            options.warmStart = ss.warmStart;
        end
        if ep
            options.epPi= ss.epPi;
            options.epBeta= ss.epBeta;
        end

        % prediction
        prob = full(pred.Yhat(miss));
        nMiss = length(prob);
        ii = floor(nMiss/2);
        [X1,X2,T,auc(iter)] = perfcurve([ones(ii,1); -ones(nMiss-ii,1);], [prob(1:ii);1-prob(ii+1:end)], 1);
        logLoss(iter) = mean(-log2(prob));
        fprintf('%d %0.4f %.4f (s2: %.3f)\n', iter, auc(iter), logLoss(iter), s2);
        binPos = [0.05:0.1:1];

        %{
        nc = hist(prob,binPos);
        subplot(212);
        bar(binPos, nc./sum(nc)*100);
        xlim([0 1]); ylim([0 100]); grid on; drawnow;
        subplot(211);
        hold on;
        plot(logLoss,'bo-'); drawnow;
        %}

        [converged, diff] = isConverged(logLoss, 1e-4, 'objFun');
        if converged
            fprintf('Converged with difference %.4f\n',diff);
            break
        end

        % estimate params
        switch method
            case 'VU'
                S = ss.S;
                s = ss.s;
                sM = ss.sM;

                S = S + 0.1*eye(size(S));
                V = hyp.loadingMat;
                vi = S\s;
                V = reshape(vi,L,M)';
                hyp.loadingMat = V;
                %hyp.mean = sM/N;

            case 'S'

            otherwise
                S = ss.S;
                s = ss.s;
                K = hyp.covMat;
                mu = mu + (K*s)/N;
                A = K + (K*(S*K))/N; %C is formed explicitely
                A = 0.5*(A + A');

                if ~nLatentDims
                    VV = A;
                else
                    A = cholKm \ (cholKm\A)';%Ctil is computed explicitely
                    A = 0.5*(A+A');
                    % eigenvalue decomposition
                    opts = struct('issym', 1, 'tol', 1e-3,'isreal',1, 'maxit', 1000, 'disp', 0);
                    [U,Lambda] = eigs(A,nLatentDims,'LM',opts);
                    % sort
                    lambda = diag(Lambda);
                    % copmute the Kernel variance
                    s2 = (trace(A) - sum(lambda))/(M-nLatentDims);
                    % PCA updates
                    gamma = max(lambda, s2) - s2;
                    W = U*diag(sqrt(gamma));
                    V = cholKm*W;
                    VV = V*V';
                end

                % update hyp
                hyp.covMat = VV + s2*(Km + Kz);
                %hyp.mean = mu;
        end
        tt(iter) = toc;
        %save initialVU mu V;

        if saveOut & ~mod(iter,5)
            fprintf('saving in %s\n', fileName);
            save(fileName, 'hyp', 'pred','logLoss','auc','tt');
        end
    end

