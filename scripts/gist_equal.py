import numpy as np, os, re, subprocess
H=os.path.expanduser('~'); n,dim,NQ=250000,960,1000
G=np.fromfile(f'{H}/gist_sub/gt.ivecs',dtype=np.int32).reshape(-1,101)[:NQ,1:51]
rng=np.random.default_rng(0)
def run(binary,a,b):
    cmd=[f'./{binary}','--dataset-path',f'{H}/gist_sub/base.fvecs','--query-path',f'{H}/gist_sub/query.fvecs','--groundtruth-path',f'{H}/gist_sub/gt.ivecs',
         '--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','50','--data-dimensionality',str(dim),'--subspace-dimensionality','6','--subspace-num','6',
         '--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/gist_sub/idx_sd6','--load-index']
    r=subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env={**os.environ,'ASC_THREADS':'1'},timeout=1800); o=r.stdout+r.stderr
    t=float(re.search(r'query time is ([0-9.]+)',o).group(1)); c=float(re.search(r'AVG_CANDIDATES ([0-9.]+)',o).group(1))
    d=np.fromfile(f'{H}/taco_results.bin',dtype=np.int32); R=d[2:].reshape(d[0],d[1])
    return np.array([len(set(R[i].tolist())&set(G[i].tolist()))/50 for i in range(NQ)]),t,c
def ci(d): bs=[d[rng.integers(0,NQ,NQ)].mean() for _ in range(2000)]; return f"{d.mean()*100:+.2f} [{np.percentile(bs,2.5)*100:+.2f}, {np.percentile(bs,97.5)*100:+.2f}]"
B=0.05; print(f"beta={B}, nominal cap {int(B*n)} candidates; one thread, Gist-250K")
print(f"{'alpha':<6}{'variant':<22}{'recall':<9}{'avg cand':<10}{'ms':<8}")
for a in (0.02,0.04,0.06,0.08):
    res={}
    for lab,bn in (('baseline (free)','taco_base_log'),('Imp1 (free)','taco_imp1_log'),('baseline (capped)','taco_base_eq'),('Imp1 (capped)','taco_imp1_eq')):
        res[lab]=run(bn,a,B); r,t,c=res[lab]; print(f"{a:<6}{lab:<22}{r.mean():<9.4f}{c:<10.0f}{t:<8.2f}",flush=True)
    print(f"      Imp1 - baseline, free  : {ci(res['Imp1 (free)'][0]-res['baseline (free)'][0])} points")
    print(f"      Imp1 - baseline, capped: {ci(res['Imp1 (capped)'][0]-res['baseline (capped)'][0])} points",flush=True)
