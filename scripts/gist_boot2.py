import numpy as np, os, subprocess, re
H=os.path.expanduser('~'); n,dim=250000,960; NQ=1000
G=np.fromfile(f'{H}/gist_sub/gt.ivecs',dtype=np.int32).reshape(-1,101)[:NQ,1:51]
def run(binary,a,b):
    cmd=[f'./{binary}','--dataset-path',f'{H}/gist_sub/base.fvecs','--query-path',f'{H}/gist_sub/query.fvecs','--groundtruth-path',f'{H}/gist_sub/gt.ivecs','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','50','--data-dimensionality',str(dim),'--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/gist_sub/idx_sd6','--load-index']
    r=subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env={**os.environ,'ASC_THREADS':'1'},timeout=900)
    t=float(re.search(r'query time is ([0-9.]+)',r.stdout+r.stderr).group(1))
    d=np.fromfile(f'{H}/taco_results.bin',dtype=np.int32); R=d[2:].reshape(d[0],d[1])
    return np.array([len(set(R[i].tolist())&set(G[i].tolist()))/50 for i in range(NQ)]), t
rng=np.random.default_rng(0)
print("alpha | base recall, ms | imp1 recall, ms | diff [95% CI] | time ratio (imp1/base)")
for a in [0.02,0.04,0.06,0.08]:
    bt=[];it=[]
    for _ in range(3):
        b_,t1=run('taco_base',a,0.05); i_,t2=run('taco_imp1',a,0.05); bt.append(t1); it.append(t2)
    d=i_-b_; bs=[d[rng.integers(0,NQ,NQ)].mean() for _ in range(2000)]
    mb,mi=np.median(bt),np.median(it)
    print(f"{a:<5} | {b_.mean():.4f}, {mb:.2f} | {i_.mean():.4f}, {mi:.2f} | {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] | {mi/mb:.2f}x",flush=True)
