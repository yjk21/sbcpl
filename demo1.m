function [] =demo1()
    clear all;

    dataDirName = '/lapmal/project/datasets/matrixFact/';
    seed = 1;
    dataName = 'sushi3a';

    jitter = 1; % epsilon

    %our parameters
    nLatentDims = 5;
    num = 10;


    % GPPE Kernel
    covfunc_t ='covSEiso_bonilla'; %user
    covfunc_x =covfunc_t; %items

    %number of users
    tsize = 200;

    % Kernel Parameters 
    % - GP full Laplace
    %   - item space kernel params
    logScaleX =  -2 ;
    logSigmaX =  -2 ;
    %   - user space kernel params
    logScaleT =  -2 ;
    logSigmaT =  -2 ;
    % - GP Item EP
    logScaleS = -2;
    logSigmaS = 1;

    % - GP Item Laplace
    logScaleSb = -2;
    logSigmaSb = 0;

    % - GPVU* 
    logScale = 0;
    logSigma = 0;

    % get data
    setSeed(seed);

    [Ytrain, Ytest, Xm, Xu] = getDataPrefLearn(dataName, dataDirName, []);

    miss = find(Ytest~=0);
    nMiss = length(miss);
    obs = find(Ytrain~=0);

    [M2,N] = size(Ytrain);
    M = sqrt(M2);
    assert(M*M == M2)

    % the GPPE method scales as (MxN)^3. thus we need to restrict N
    idx = randsample(N, tsize);
    YtrainSmall = Ytrain(:,idx);
    YtestSmall  = Ytest(:,idx);
    XuTemp = Xu(idx,:);

    % convert Ytrain to gppe format (cell array)
    [train_pairs, ind_x, ind_t, idx_global, idx_global_1, idx_global_2] = gppeConvertData(tsize, YtrainSmall, M);

    % run S and Sb
    [logLoss] = gpitem('ppcaJak', dataName, nLatentDims, 1, logScaleS, logSigmaS,seed, sparse(YtrainSmall), sparse(YtestSmall), Xm, XuTemp);
    errS = logLoss(end);

    % inference
    theta = [0, 0, logScaleSb, logSigmaSb, 1];
    [fhat, Kx, Kinv, W, L] = approx_gppe_laplace_fast_nouser( covfunc_t, covfunc_x, theta, XuTemp, Xm, train_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, tsize, M, jitter);

    prob_mm = gppePredict(tsize, YtestSmall,fhat, Kx, Kinv, W, L, covfunc_t, covfunc_x, theta, XuTemp, Xm, idx_global, ind_t, ind_x, M, jitter);
    errSb = mean(-log2(prob_mm));


    % run GP full
    theta = [logScaleT, logSigmaT, logScaleX, logSigmaX, 0];
    [fhat, Kx, Kinv, W, L] = approx_gppe_laplace_fast(...
    covfunc_t, ...
    covfunc_x, ...
    theta, ...
    XuTemp, ...
    Xm, ...
    train_pairs, ...
    idx_global, idx_global_1, idx_global_2, ...
    ind_t, ...
    ind_x, ...
    tsize, ...
    M, jitter);

    prob_mm = gppePredict(tsize, YtestSmall, fhat, Kx, Kinv, W, L, covfunc_t, covfunc_x, theta, XuTemp, Xm, idx_global, ind_t, ind_x, M, jitter);

    errB = mean(-log2(prob_mm));

    % Run VU
    [logLoss, s2, tt] = gpvu('VU', dataName, nLatentDims, 0,...
    0, seed, sparse(YtrainSmall), sparse(YtestSmall), Xm, XuTemp);
    errVU = logLoss(end);

    % run GPVU Jaakkola
    [logLoss, s2GPVUJ, tt] = gpvu('ppcaJak', dataName, nLatentDims, 1,...
    logScale, seed, sparse(YtrainSmall), sparse(YtestSmall), Xm, XuTemp);
    errZJak = logLoss(end);

    % run GPVU EP
    [logLoss, s2GPVUE, tt] = gpvu('ppcaEp', dataName, nLatentDims, 1,...
    logScale, seed, sparse(YtrainSmall), sparse(YtestSmall), Xm, XuTemp);



function [train_pairs, ind_x, ind_t, idx_global, idx_global_1, idx_global_2] = gppeConvertData(tsize, Ytrain, M)
    train_pairs = cell(tsize,1);    % each entry corresponds to a user
    for n = 1:tsize
        obsIdx = find( Ytrain(:,n) ~=0 );
        [pos, neg] = ind2sub([M, M], obsIdx);
        train_pairs{n} = [pos neg];
    end
    % some magic tricks
    [idx_global_1, idx_global_2] = compute_global_index(train_pairs, M);
    idx_global    = unique([idx_global_1; idx_global_2]);
    [ind_x,ind_t] = ind2sub([M, tsize], idx_global); % idx of seen points and tasks

function [prob_mm] = gppePredict(tsize, Ytest, fhat, Kx, Kinv, W, L, covfunc_t, covfunc_x, theta, XuTemp, Xm, idx_global, ind_t, ind_x, M, jitter)
    prob_mm = [];
    for n = 1:tsize         % Ytest should be M*M x N
        obsIdx = find(Ytest(:,n) ~=0 );
        [pidxT, nidxT] = ind2sub([M, M], obsIdx);
        if ~isempty(pidxT)
            for ii = 1:length(pidxT)
                [p,mustar] = predict_gppe_laplace(...
                covfunc_t, ...
                covfunc_x, ...
                theta,...
                fhat, ...
                Kx, ...
                Kinv, ...
                W, ...
                L,...
                XuTemp, ...
                Xm, ...
                idx_global, ...
                ind_t, ...
                ind_x, ...
                XuTemp(n,:), ...
                [pidxT(ii) nidxT(ii)], jitter);
                prob_mm = [prob_mm; p];
            end
        end
    end


