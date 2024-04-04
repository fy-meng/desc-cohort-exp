include "globals.mzn";

int: k;  % num clusters
int: n;  % num samples
int: f;  % num features
int: r;  % num tags

int: alpha; 
int: beta;

int: q;

set of int: NS = 1..n;
set of int: FS = 1..f;
set of int: RS = 1..r;
set of int: KS = 1..k;

array[NS,FS] of float: X;      % samples
array[NS,RS] of bool: D;       % tags

array[NS] of var 1..k: G;       % label
array[KS,RS] of var bool: S;   % description

constraint forall(c in KS)(
  count(G, c) >= 1
);
constraint seq_precede_chain(G);

constraint forall(i in NS)(
  sum(p in RS)(S[G[i],p] * (1 - D[i,p])) <= alpha
);
constraint forall(c in KS, p in RS)(
  (S[c,p] = 1) <-> (sum(i in NS)(if G[i] = c then 1 - D[i,p] else 0 endif) <= beta)
);

var int: descriptiveness = min(c in KS)(
  sum(p in RS)(S[c,p])
);
constraint descriptiveness >= q;

var float: compactness = sum(i,j in NS where i < j)(
  if G[i] = G[j] then sum(t in FS)((X[i,t] - X[j,t])^2) else 0 endif
) / n / f;

solve minimize compactness;
