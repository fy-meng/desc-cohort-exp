int: k;  % num clusters
int: n;  % num samples
int: f;  % num features
int: r;  % num tags

int: alpha = 2; 
int: beta = 0;

set of int: NS = 1..n;
set of int: FS = 1..f;
set of int: RS = 1..r;
set of int: KS = 1..k;

array[NS,FS] of float: X;      % samples
array[NS,RS] of bool: D;       % tags

array[NS,KS] of var bool: Z;   % assignment
array[KS,RS] of var bool: S;   % description

constraint forall(i in NS)(sum(c in KS)(Z[i,c]) = 1);
constraint forall(c in KS)(sum(i in NS)(Z[i,c]) >= 1);
constraint forall(c in KS)(sum(p in RS)(S[c,p]) >= 1);

constraint forall(c in KS, i in NS)(
  sum(p in RS)( (S[c,p] + Z[i,c] - 1) * (1 - D[i,p]) ) <= alpha
);
constraint forall(c in KS, p in RS)(
  sum(i in NS)( (S[c,p] + Z[i,c] - 1) * (1 - D[i,p]) ) <= beta
);
constraint forall(c in KS, p in RS)(
  (n + 1) * S[c,p] >= 1 + beta - sum(i in NS)( Z[i,c] * (1 - D[i,p]) )
);
  
var float: compactness = sum(i,j in NS where i < j)(
  sum(c in KS)(Z[i,c] * Z[j,c]) * sum(t in FS)((X[i,t] - X[j,t])^2)
) / n / f;
var float: descriptiveness = min(c in KS)(sum(p in RS)(S[c,p]));

solve minimize compactness;

