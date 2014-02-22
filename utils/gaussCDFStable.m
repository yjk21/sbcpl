% But emti didn't write it
%% Util: Gauss CDF, implemented using error function
function [p,lp] = GaussCDFStable(y,f)
  if numel(y)>0, yf = y.*f; else yf = f; end     % product of latents and labels
  p  = (1+erf(yf/sqrt(2)))/2;                                       % likelihood
  if nargout>1, lp = logphi(yf,p); end                          % log likelihood

