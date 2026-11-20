import numpy as np, os, re, subprocess, hnswlib
H=os.path.expanduser('~'); D96=f'{H}/deep96_subsets'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
DS={'Deep96-1M':dict(base=f'{D96}/base_1000000.fvecs',q=f'{D96}/queries.fvecs',g=f'{D96}/gt_1000000.ivecs',n=1000000,d=96,s='deep96_1m',ns=6,K=50,ef=100,a=0.04,b=0.01),
    'Gist-250K':dict(base=f'{H}/gist_sub/base.fvecs',q=f'{H}/gist_sub/query.fvecs',g=f'{H}/gist_sub/gt.ivecs',n=250000,d=960,s='gist_250k',ns=8,K=50,ef=200,a=0.08,b=0.01)}
NQ=1000; K=50
def fv(p,n,d): return np.ascontiguousarray(np.fromfile(p,dtype=np.float32,count=n*(d+1)).reshape(n,d+1)[:,1:])
def lid_mle(dist):
    r=dist/dist[-1]; r=r[:-1]; r=r[r>0]
    return -1.0/np.mean(np.log(r))
def suco_ids(c):
    S=f'{H}/suco_data/{c["s"]}'; sd=c['d']//c['ns']; idx=f'{H}/tune_idx_{c["s"]}_{c["ns"]}_{c["K"]}'
    res=f'{H}/suco_results.bin'
    if os.path.exists(res): os.remove(res)
    cmd=['./suco_base','--dataset-path',f'{S}/base.bin','--query-path',f'{S}/query.bin','--groundtruth-path',f'{S}/gt.bin','--dataset-size',str(c['n']+100),'--query-size',str(NQ),'--k-size',str(K),'--data-dimensionality',str(c['d']),'--subspace-dimensionality',str(sd),'--subspace-num',str(c['ns']),'--candidate-ratio',str(c['b']),'--collision-ratio',str(c['a']),'--kmeans-num-centroid',str(c['K']),'--kmeans-num-iters','2','--index-path',idx,'--load-index']
    subprocess.run(cmd,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=7200)
    a=np.fromfile(res,dtype=np.int32); return a[2:].reshape(a[0],a[1])
def rank(x): return np.argsort(np.argsort(x)).astype(float)
for name,c in DS.items():
    X=fv(c['base'],c['n'],c['d']); Q=fv(c['q'],NQ,c['d']); G=np.fromfile(c['g'],dtype=np.int32).reshape(-1,101)[:NQ,1:K+1]
    lid=np.array([lid_mle(np.sort(np.linalg.norm(X[G[i]]-Q[i],axis=1))) for i in range(NQ)])
    h=hnswlib.Index(space='l2',dim=c['d']); h.init_index(max_elements=c['n'],ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_num_threads(1); h.set_ef(c['ef'])
    hn=[h.knn_query(v[None],k=K)[0][0] for v in Q]
    rec={'HNSW':np.array([len(set(hn[i].tolist())&set(G[i].tolist()))/K for i in range(NQ)])}
    try:
        sc=suco_ids(c); rec['SuCo']=np.array([len(set(sc[i].tolist())&set(G[i].tolist()))/K for i in range(NQ)])
    except Exception as e: print('SuCo failed:',e)
    print(f"\n##### {name}: LID (MLE over the {K} nearest neighbours): median {np.median(lid):.1f}, 10th to 90th percentile {np.percentile(lid,10):.1f} to {np.percentile(lid,90):.1f}")
    q=np.quantile(lid,[0.25,0.5,0.75]); bins=np.digitize(lid,q)
    print(f"{'Method':<8}{'Spearman rho':<14}"+''.join(f"{'LID quartile '+str(i+1):<16}" for i in range(4)))
    for m,r in rec.items():
        rho=np.corrcoef(rank(lid),rank(r))[0,1]
        print(f"{m:<8}{rho:<14.3f}"+''.join(f"{r[bins==i].mean():<16.4f}" for i in range(4)))
    print("(quartile 1 = lowest LID = easiest queries; recall@%d, mean per quartile)"%K)
