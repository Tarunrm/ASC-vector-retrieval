import numpy as np, os, re, subprocess, statistics
H=os.path.expanduser('~'); S=f'{H}/suco_data'
CFG=[('deep96_1m',1000000,96,6,16),('sift_1m',1000000,128,8,16),('gist_250k',250000,960,8,120)]
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
rng=np.random.default_rng(0)
def run(binary,name,n,d,ns,sd,a,b):
    cmd=[f'./{binary}','--dataset-path',f'{S}/{name}/base.bin','--query-path',f'{S}/{name}/query.bin','--groundtruth-path',f'{S}/{name}/gt.bin',
         '--dataset-size',str(n+100),'--query-size','1000','--k-size','50','--data-dimensionality',str(d),'--subspace-dimensionality',str(sd),
         '--subspace-num',str(ns),'--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2',
         '--index-path',f'{H}/sv_idx_suco_{name}','--load-index']
    o=subprocess.run(cmd,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=3600); o=o.stdout+o.stderr
    t=float(re.search(r'query time is ([0-9.]+)',o).group(1))
    r=np.fromfile(f'{H}/suco_results.bin',dtype=np.int32); R=r[2:].reshape(r[0],r[1]); return R,t
print("dataset     alpha beta  base    imp1    diff [95% CI]            time ratio")
for name,n,d,ns,sd in CFG:
    G=np.fromfile(f'{S}/{name}/gt.bin',dtype=np.int64).reshape(-1,50)[:1000]
    for a in [0.02,0.04,0.08]:
        b=0.01; bt=[];it=[]
        for _ in range(3):
            Rb,t1=run('suco_base',name,n,d,ns,sd,a,b); Ri,t2=run('suco_imp1',name,n,d,ns,sd,a,b); bt.append(t1); it.append(t2)
        rb=np.array([len(set(Rb[i].tolist())&set(G[i].tolist()))/50 for i in range(1000)])
        ri=np.array([len(set(Ri[i].tolist())&set(G[i].tolist()))/50 for i in range(1000)])
        df=ri-rb; bs=[df[rng.integers(0,1000,1000)].mean() for _ in range(2000)]
        print(f"{name:<11} {a:<5} {b:<5} {rb.mean():.4f}  {ri.mean():.4f}  {df.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  {statistics.median(it)/statistics.median(bt):.2f}x",flush=True)
