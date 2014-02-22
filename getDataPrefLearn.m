function [Ytrain, Ytest, Xm, Xu] = getDataPrefLearn(name, dirName, options)
% We do the following:
% sort d's and n's
% subsample d's and n's
% transform the data
% remove d's and n's with no ratings
% create one missing entry
% return (ytrain, ytest)

  [plotHist, plotMI, ratio] = myProcessOptions(options, ...
      'plotHist',0,'plotMI',0, 'ratio', 0.7);

  fprintf('Loading %s data\n', name);
  switch name
      case 'synth0'
          % synthetic data
          mu = [1 0 -1];
          K = eye(3);
          eta = mvnrnd(mu, K, 1000);
          B = [0 0 0; -1 1 0; -1 0 1;...
              1 -1 0; 0 0 0; 0 -1 1;...
              1 0 -1; 0 1 -1; 0 0 0;];
          Y = B*eta';
          Ytrain = Y>eps;
          Ytest = zeros(size(Ytrain));
          Xm = [];
          Xu = [];

      case 'bonillaToyData'
        N = 100;     % Total Number of users (including test users)
        M = 10;     % Number of items
        covfunc_t = 'covSEard_bonilla';
        covfunc_x = 'covSEard_bonilla';
        D_x = 3;   % dimensionality of the item features
        D_t = 2;   % Dimesionality of user feaures
        sigma = 0.01; % Noise parameter  
        %% Generating features and latent functions
        EPSILON       = sigma*randn;
        Mtrain_ratio   = 1; % How many training pairs to use
        %t = 5*(rand(N,D_t)-0.5);
        t = 15*(rand(N,D_t)-0.5);
        x = 15*(rand(M,D_x)-0.5);
        logtheta_x = zeros(D_x+1,1);
        logtheta_t = zeros(D_t+1,1);
        theta = [logtheta_t; logtheta_x; log(sigma)];
        Kt = feval(covfunc_t, logtheta_t, t);
        Kx = feval(covfunc_x, logtheta_x, x);
        cond(Kt)
        cond(Kx)
        pause

        % K is the kronecker product Kf (x) Kx
        K  = kron(Kt, Kx);

        % now genereates the targets Y: gaussian mean 0 cov K
        % we sample this multivariate gaussian distribution N(mu,K)
        n  = N*M;
        mu = zeros(n,1);
        f  = mu + chol(K)'*randn(n,1);
        F  = reshape(f,M,N);

        idx_pairs =  combnk(1:M, 2); % all possible pairs
        Y = ( F(idx_pairs(:,1) ,:) - F(idx_pairs(:,2),:) ) > EPSILON;
            
        % For each user we create a cell that contains the ordered pairs
        all_pairs = cell(1,M);
        for j = 1 : N
            tmp_pairs = idx_pairs;
            idx_0 = find(Y(:,j) == 0); % indices in reverse order
            tmp_pairs(idx_0,:) = fliplr(idx_pairs(idx_0,:));
            all_pairs{j} = tmp_pairs;
        end

        %% Assigning training and testing data
        Ntrain = N;
        test_idx = N;
        Mpairs_train = floor(size(idx_pairs,1)*Mtrain_ratio);
        train_pairs = get_training_pairs( all_pairs, Ntrain, Mpairs_train);

        %convert back to my format
        Ytrain = zeros(M^2,N);
        for n = 1:N
          i = train_pairs{n}(:,1);
          j = train_pairs{n}(:,2);
          ind = sub2ind([M M], i, j);
          Ytrain(ind,n) = 1;
        end

        % create test-train pairs
        nMiss = 1;
        dLowerLimit = 2;
        [Ytrain,Ytest] = punchHoles(Ytrain, nMiss, dLowerLimit);

        Ytrain = sparse(Ytrain);
        Ytest = sparse(Ytest);
        Xu = t;
        Xm = x;
        
      case 'synth1'
          % synthetic data
          M = 5;
          % first group 1>2>...>M
          Y1 = triu(ones(M),1);
          % second group 1<2<...<M
          Y2 = Y1';
          %Y2(1,:) = Y2(:,1);
          %Y2(:,1) = 0;
          Y = [repmat(Y1(:),1, 10) repmat(Y2(:), 1, 10)];
          % test leave one preference out
          Ytrain = zeros(size(Y));;
          Ytest = zeros(size(Y));;
          for n = 1:size(Y,2)
              On = find(Y(:,n));
              idx = randperm(length(On));
              d = On(idx(1));
              Ytest(d,n) = 1;
              d = On(idx(2:M));
              Ytrain(d,n) = 1;
          end
          Xm = [];
          Xu = [];

       case 'sushi3a'
          nMiss = 1; % #of missing values per user
          dLowerLimit = 2; % #comparison user should have to get a test data point
          
          load(sprintf('%s/sushi3/sushi3a_rankBasedPref.mat',dirName));
          Xm = itemFeatures;
          Xu = userFeatures;
          [M] = size(Xm,1);
          [N] = size(Xu,1);
          Y = zeros(M*M,N);
          for n = 1:N
              ind = sub2ind([M,M], itemIdxPos(:,n), itemIdxNeg(:,n));
              idx = randperm(length(ind));
              ind = ind(idx(1:3));
              Y(ind,n) = 1;
          end
          
          [Y] = removeUniformativeDims(Y);
          [Ytrain,Ytest] = punchHoles(Y, nMiss, dLowerLimit);
          Ytrain = sparse(Ytrain);
          Ytest = sparse(Ytest);
         
          Ytrain = Ytrain(:,1:1000);
          Ytest= Ytest(:,1:1000);

      case 'sushi3a_emti'

        sushiA = load(sprintf('%ssushi3/sushi3a_rankBasedPref.mat',dirName));
        Xm = sushiA.itemFeatures;
        Xu = sushiA.userFeatures;
        [M] = size(Xm,1);
        [N] = size(Xu,1);
        Y = zeros(M*M,N);
        Ytest = zeros(M*M,N);

        % create sets Io>Jo for training and Ih>Jh for testing
        for n = 1:N
            ranks = sushiA.rankingA(:,n);
            % take high ranked sushi
            I = ranks(1:5);
            idx = randperm(length(I));
            Io = I(idx(1:3));
            Ih = I(idx(4:5));
            % take low ranked sushi
            J = ranks(6:10);
            idx = randperm(length(J));
            Jo = J(idx(1:3));
            Jh = J(idx(4:5));
            % train set is Io>Jo
            [i,j] = createAllPairs(Io,Jo);
            ind = sub2ind([M,M], i, j);
            Y(ind,n) = 1;
            % test set is Ih>Jh
            [i,j] = createAllPairs(Ih,Jh);
            ind = sub2ind([M,M], i, j);
            Ytest(ind,n) = 1;
        end
        
        nonZeros = sum(Y~=0,1);
        keep = (nonZeros~=0);
        
        if ~all(keep)
            Y =  Y(:,keep);
            Ytest = Ytest(:,keep);
        end
        
        Y= Y(:,1:20);
        Ytest= Ytest(:,1:20);

        Ytrain = sparse(Y);
        Ytest = sparse(Ytest);

      case 'sushi3b'
          nMiss = 1; % #of missing values per user
          dLowerLimit = 2; % #comparison user should have to get a test data point
          
          load(sprintf('%s/sushi3/sushi3b_rankBasedPref.mat',dirName));
          Xm = itemFeatures;
          Xu = userFeatures;
          [M] = size(Xm,1);
          [N] = size(Xu,1);
          Y = zeros(M*M,N);
          for n = 1:N
              ind = sub2ind([M,M], itemIdxPos(:,n), itemIdxNeg(:,n));
              idx = randperm(length(ind));
              ind = ind(idx(1:10));
              Y(ind,n) = 1;
              count(n) = length(ind);
          end
          
          [Y] = removeUniformativeDims(Y);
          [Ytrain,Ytest] = punchHoles(Y, nMiss, dLowerLimit);
          Ytrain = sparse(Ytrain);
          Ytest = sparse(Ytest);

    case 'sushi3b_emti'
        sushiB = load(sprintf('%ssushi3/sushi3b_rankBasedPref.mat',dirName));
        
        Xm = sushiB.itemFeatures;
        Xu = sushiB.userFeatures;
        [M] = size(Xm,1);
        [N] = size(Xu,1);
        Y = zeros(M*M,N);
        Ytest = zeros(M*M,N);

        % create sets Io>Jo for training and Ih>Jh for testing
        for n = 1:N
            ranks = sushiB.rankingB(:,n);
            % take high ranked sushi
            I = ranks(1:5);
            idx = randperm(length(I));
            Io = I(idx(1:3));
            Ih = I(idx(4:5));
            % take low ranked sushi
            J = ranks(6:10);
            idx = randperm(length(J));
            Jo = J(idx(1:3));
            Jh = J(idx(4:5));
            % train set is Io>Jo
            [i,j] = createAllPairs(Io,Jo);
            ind = sub2ind([M,M], i, j);
            Y(ind,n) = 1;
            % test set is Ih>Jh
            [i,j] = createAllPairs(Ih,Jh);
            ind = sub2ind([M,M], i, j);
            Ytest(ind,n) = 1;
        end
        
        nonZeros = sum(Y~=0,1);
        keep = (nonZeros~=0);
        
        if ~all(keep)
            Y =  Y(:,keep);
            Ytest = Ytest(:,keep);
        end
        
        Ytrain = sparse(Y);
        Ytest = sparse(Ytest);

      case 'sushi3b_fused'
          nMiss = 1; % #of missing values per user
          dLowerLimit = 2; % #comparison user should have to get a test data point
          
          load(sprintf('sushi3b_rankBasedPref.mat'));
          Xm = itemFeatures;
          Xu = userFeatures;
          [M] = size(Xm,1);
          [N] = size(Xu,1);
          Y = zeros(M*M,N);
          YtrainX = 6-ratings;
          nZero = 10;
          for n = 1:N
              % implicit ratings
              ind = sub2ind([M,M], itemIdxPos(:,n), itemIdxNeg(:,n));
              idx = randperm(length(ind));
              ind = ind(idx(1:10));
              Y(ind(1:3),n) = 1;
              % remove some of the explicit ratings
              ind = find(YtrainX(:,n)~=0);
              %full([ind YtrainX(ind,n) rankingB(:,n)])
              %pause
              ll = length(ind);
              idx = randperm(ll);
              ind = ind(idx(1:ll-nZero));
              YtrainX(ind,n) = 0;
          end
          
          [Y] = removeUniformativeDims(Y);
          [YtrainI,Ytest] = punchHoles(Y, nMiss, dLowerLimit);
          
          %{
          for n = 1:N
              yt = Ytest(:,n);
              Ot = find(yt);
              [i,j] = ind2sub([M,M], Ot);
              
              YtrainX(i,n) = 0;
              YtrainX(j,n) = 0;
          end
          %}
          Ytrain.implicit = sparse(YtrainI);
          Ytrain.explicit = sparse(YtrainX);
          Ytest = sparse(Ytest);
          
      case 'sushi3bvsa'
          nMiss = 1; % #of missing values per user
          dLowerLimit = 2; % #comparison user should have to get a test data point
          
          sushiA = load(sprintf('%ssushi3/sushi3a_rankBasedPref.mat',dirName));
          sushiB = load(sprintf('%ssushi3/sushi3b_rankBasedPref.mat',dirName));
          
          %we use the pairs from ranking A as test data
          %hence we need to remove any pairs in the training data resulting
          %from ranking A
          
          Xm = sushiB.itemFeatures;
          Xu = sushiB.userFeatures;
          [M] = size(Xm,1);
          [N] = size(Xu,1);
          Y = zeros(M*M,N);
          Ytest = zeros(M*M,N);
          for n = 1:N
              indB = sub2ind([M,M], sushiB.itemIdxPos(:,n), sushiB.itemIdxNeg(:,n));
              Y(indB,n) = 1;
              ranks = sushiA.rankingA(:,n);
              S1 = ranks(8:10);
              S2 = ranks(1:3);
              % stupid way of creating all pairs of set 1 and 2
              iPos = []; iNeg= [];
              for t = 1:length(S1)
                  for tt = 1:length(S2)
                      iPos = [iPos; S1(t)];
                      iNeg = [iNeg; S2(tt)];
                  end
              end
              indA = sub2ind([M,M], iPos, iNeg);
              %indA = sub2ind([M,M], sushiA.itemIdxPos(:,n), sushiA.itemIdxNeg(:,n));
              Y(indA,n) = 0;
              Ytest(indA,n) = 1;
          end
          
          nonZeros = sum(Y~=0,1);
          keep = (nonZeros~=0);
          
          if ~all(keep)
              Y =  Y(:,keep);
              Ytest = Ytest(:,keep);
          end
          
          Ytrain = sparse(Y);
          Ytest = sparse(Ytest);
          %         [Y] = removeUniformativeDims(Y);
          %         [Ytrain,Ytest] = punchHoles(Y, nMiss, dLowerLimit);
          
      otherwise
          error('no such name');
  end

  % check for dimensions with no observations
  %if ~isempty(find(sum(Ytrain~=0,1)==0)) || ~isempty(find(sum(Ytrain~=0,2)==0))
  %    error('rows or columns with no observations');
  %end

  fprintf('#items %d, #users %d\n',sqrt(size(Ytrain,1)), size(Ytrain,2));
  fprintf('#train preferences %d\n',full(sum(Ytrain(:)~=0)));
  fprintf('#test preferences %d\n', full(sum(Ytest(:)~=0)));

  function [Y,Xm,Xu] = sortData(Y,varargin)

  if nargin>1;
      Xm = varargin{1};
      if nargin>2;
          Xu = varargin{2};
      else
          Xu = [];
      end
  else
      Xm = [];
      Xu=[];
  end

  % sort n's according to #of ratings
  On = sum(Y~=0);
  [junk,idx] = sort(On,2,'descend');
  Y = Y(:,idx);
  if ~isempty(Xu)
      Xu = Xu(idx,:);
  end
  % sort d's according to #of ratings
  Od = sum(Y~=0,2);
  [junk,idx] = sort(Od,1,'descend');
  Y = Y(idx,:);
  if ~isempty(Xm)
      Xm = Xm(idx,:);
  end

  function [Y,Xm,Xu] = removeUniformativeDims2(Y,varargin)
  if nargin>1; Xm = varargin{1};
      if nargin>2; Xu = varargin{2}; else Xu = []; end
  else Xm = [];Xu=[]; end

  % remove n's with no ratings
  On = sum(Y~=0);
  idx = find(On ~=0);
  Y = Y(:,idx);
  % remove d's with no ratings
  Od = sum(Y~=0,2);
  idx = find(Od ~= 0);
  Y = Y(idx,:);
  if ~isempty(Xm)
      Xm = Xm(idx,:);
  end

  function [Y,Xm,Xu] = removeUniformativeDims(Y,varargin)

  if nargin>1; Xm = varargin{1};
      if nargin>2; Xu = varargin{2}; else Xu = []; end
  else Xm = [];Xu=[]; end

  % remove n's with no ratings
  On = sum(Y~=0);
  idx = find(On ~=0);
  Y = Y(:,idx);
  %{
  % remove d's with no ratings
  Od = sum(Y~=0,2);
  idx = find(Od ~= 0);
  Y = Y(idx,:);
  if ~isempty(Xm)
  Xm = Xm(idx,:);
  end
  %}


  function [Ytrain, Ytest]  = punchHoles(Y, nMiss, dLowerLimit)

  [D N] = size(Y);
  Ytrain = Y;
  %Ytest = sparse(D,N);
  num = sum(Y~=0,2);
  dd = [];
  nn = [];
  yy = [];
  for n = 1:N
      On = find(Y(:,n)~=0);
      if length(On)>dLowerLimit
          numMiss = nMiss;
          ind = unidrnd(length(On),[numMiss 1]);
          d = On(ind);
          dd = [dd; d];
          nn = [nn; repmat(n,numMiss,1)];
          yy = [yy; 2*Y(d,n)-1];
          %Ytest(d,n) = Y(d,n);
          %Ytrain(d,n) = 0;
      end
  end

  Ytest = sparse(dd,nn,yy,D,N);
  Ytrain(sub2ind([D N], dd, nn)) = 0;

%% get training pairs
function train_pairs = get_training_pairs( all_pairs, Mtrain, Ntrain)
all_pairs = all_pairs(1:Mtrain);
train_pairs = cell(Mtrain,1);
for j = 1 : Mtrain
    pairs = all_pairs{j};
    idx   = randperm(size(pairs,1));
    idx   = idx(1:Ntrain);
    train_pairs{j} = pairs(idx, :);
end


