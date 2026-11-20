import numpy as np, os, subprocess, shutil
H=os.path.expanduser('~'); n,dim=250000,960; NQ=1000
G=np.fromfile(f'{H}/gist_sub/gt.ivecs',dtype=np.int32).reshape(-1,101)[:NQ,1:51]
def run(binary,a,b):
    cmd=[f'./{binary}','--dataset-path',f'{H}/gist_sub/base.fvecs','--query-path',f'{H}/gist_sub/query.fvecs','--groundtruth-path',f'{H}/gist_sub/gt.ivecs','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','50','--data-dimensionality',str(dim),'--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/gist_sub/idx_sd6','--load-index']
    subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env={**os.environ,'ASC_THREADS':'1'},timeout=900)
    d=np.fromfile(f'{H}/taco_results.bin',dtype=np.int32); R=d[2:].reshape(d[0],d[1])
    return np.array([len(set(R[i].tolist())&set(G[i].tolist()))/50 for i in range(NQ)])
rng=np.random.default_rng(0)
print("alpha  base    imp1    diff    95% CI (paired bootstrap)")
for a in [0.02,0.04,0.06,0.08]:
    b_=run('taco_base',a,0.05); i_=run('taco_imp1',a,0.05); d=i_-b_
    bs=[d[rng.integers(0,NQ,NQ)].mean() for _ in range(2000)]
    print(f"{a:<6} {b_.mean():.4f}  {i_.mean():.4f}  {d.mean():+.4f}  [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]",flush=True)
