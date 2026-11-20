import numpy as np, os, re, subprocess, hnswlib
H=os.path.expanduser('~'); D96=f'{H}/deep96_subsets'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
DS={'Deep96-1M':dict(base=f'{D96}/base_1000000.fvecs',q=f'{D96}/queries.fvecs',g=f'{D96}/gt_1000000.ivecs',n=1000000,d=96,s='deep96_1m',ns=6,K=50,ef=(100,200)),
    'Gist-250K':dict(base=f'{H}/gist_sub/base.fvecs',q=f'{H}/gist_sub/query.fvecs',g=f'{H}/gist_sub/gt.ivecs',n=250000,d=960,s='gist_250k',ns=8,K=50,ef=(200,400))}
KS=(10,20,30,40,50); SUCO=((0.04,0.01),(0.08,0.01),(0.08,0.05)); NQ=1000
def fv(p,n,d): return np.ascontiguousarray(np.fromfile(p,dtype=np.float32,count=n*(d+1)).reshape(n,d+1)[:,1:])
def suco_ids(c,a,b):
    S=f'{H}/suco_data/{c["s"]}'; sd=c['d']//c['ns']; idx=f'{H}/tune_idx_{c["s"]}_{c["ns"]}_{c["K"]}'; res=f'{H}/suco_results.bin'
    if os.path.exists(res): os.remove(res)
    cmd=['./suco_base','--dataset-path',f'{S}/base.bin','--query-path',f'{S}/query.bin','--groundtruth-path',f'{S}/gt.bin','--dataset-size',str(c['n']+100),'--query-size',str(NQ),'--k-size','50','--data-dimensionality',str(c['d']),'--subspace-dimensionality',str(sd),'--subspace-num',str(c['ns']),'--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid',str(c['K']),'--kmeans-num-iters','2','--index-path',idx,'--load-index']
    o=subprocess.run(cmd,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=7200); o=o.stdout+o.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    r=np.fromfile(res,dtype=np.int32); return r[2:].reshape(r[0],r[1]),float(m.group(1)),float(t.group(1))
def rk(L,G,k): return np.mean([len(set(L[i][:k].tolist())&set(G[i,:k].tolist()))/k for i in range(NQ)])
print(f"{'Dataset':<10}{'Method':<28}"+''.join(f"{'R@'+str(k):<9}" for k in KS)+"binary R@50 / ms")
for name,c in DS.items():
    X=fv(c['base'],c['n'],c['d']); Q=fv(c['q'],NQ,c['d']); G=np.fromfile(c['g'],dtype=np.int32).reshape(-1,101)[:NQ,1:]
    h=hnswlib.Index(space='l2',dim=c['d']); h.init_index(max_elements=c['n'],ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_num_threads(1)
    for ef in c['ef']:
        h.set_ef(ef); L=[h.knn_query(v[None],k=50)[0][0] for v in Q]
        print(f"{name:<10}{'HNSW ef='+str(ef):<28}"+''.join(f"{rk(L,G,k):<9.4f}" for k in KS),flush=True)
    for a,b in SUCO:
        ids,rb,t=suco_ids(c,a,b)
        print(f"{name:<10}{'SuCo a='+str(a)+' b='+str(b):<28}"+''.join(f"{rk(ids,G,k):<9.4f}" for k in KS)+f"{rb:.4f} / {t} ms",flush=True)
