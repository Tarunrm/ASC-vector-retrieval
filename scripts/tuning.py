import numpy as np, os, re, subprocess, time, hnswlib
H=os.path.expanduser('~'); D96=f'{H}/deep96_subsets'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
DS={'Deep96-1M':dict(base=f'{D96}/base_1000000.fvecs',q=f'{D96}/queries.fvecs',g=f'{D96}/gt_1000000.ivecs',n=1000000,d=96,s='deep96_1m'),
    'SIFT-1M':dict(base=f'{H}/sift/sift_base.fvecs',q=f'{H}/sift/sift_query.fvecs',g=f'{H}/sift/sift_groundtruth.ivecs',n=1000000,d=128,s='sift_1m'),
    'Gist-250K':dict(base=f'{H}/gist_sub/base.fvecs',q=f'{H}/gist_sub/query.fvecs',g=f'{H}/gist_sub/gt.ivecs',n=250000,d=960,s='gist_250k')}
def fv(p,n,d): return np.ascontiguousarray(np.fromfile(p,dtype=np.float32,count=n*(d+1)).reshape(n,d+1)[:,1:])
print("##### HNSW: effect of M and efConstruction (build on 8 threads, queries on one thread) #####",flush=True)
HN={'Deep96-1M':[(8,100),(16,100),(16,200),(16,400),(32,200),(32,400),(48,200)],'Gist-250K':[(16,200),(32,200),(48,200),(32,400)]}
for name,cfgs in HN.items():
    c=DS[name]; X=fv(c['base'],c['n'],c['d']); Q=fv(c['q'],1000,c['d']); G=np.fromfile(c['g'],dtype=np.int32).reshape(-1,101)[:1000,1:51]
    for M,efc in cfgs:
        t=time.time(); h=hnswlib.Index(space='l2',dim=c['d']); h.init_index(max_elements=c['n'],ef_construction=efc,M=M); h.set_num_threads(8); h.add_items(X); bt=time.time()-t; h.set_num_threads(1)
        for ef in (50,100,200,400,800):
            h.set_ef(ef); t=time.time(); L=[h.knn_query(v[None],k=50)[0][0] for v in Q]; ms=(time.time()-t)/1000*1000
            rc=np.mean([len(set(L[i].tolist())&set(G[i].tolist()))/50 for i in range(1000)])
            print(f"{name:<10} M={M:<3} efC={efc:<4} build {bt:6.0f}s  ef={ef:<4} recall {rc:.4f}  {ms:.3f} ms",flush=True)
        del h
def suco(name,ns,K,a,b,nq,load):
    c=DS[name]; S=f'{H}/suco_data/{c["s"]}'; sd=c['d']//ns; idx=f'{H}/tune_idx_{c["s"]}_{ns}_{K}'
    cmd=['./suco_base','--dataset-path',f'{S}/base.bin','--query-path',f'{S}/query.bin','--groundtruth-path',f'{S}/gt.bin','--dataset-size',str(c['n']+100),'--query-size',str(nq),'--k-size','50','--data-dimensionality',str(c['d']),'--subspace-dimensionality',str(sd),'--subspace-num',str(ns),'--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid',str(K),'--kmeans-num-iters','2','--index-path',idx]+(['--load-index'] if load else [])
    o=subprocess.run(cmd,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=7200); o=o.stdout+o.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    return (float(m.group(1)),float(t.group(1))) if m and t else (None,None)
print("\n##### SuCo: effect of the number of subspaces and K (one thread) #####",flush=True)
SC={'Deep96-1M':[(4,50),(6,50),(8,50),(12,50),(16,50),(6,100)],'SIFT-1M':[(4,50),(8,50),(16,50),(8,100)],'Gist-250K':[(4,50),(8,50),(16,50),(8,100)]}
for name,cfgs in SC.items():
    for ns,K in cfgs:
        suco(name,ns,K,0.04,0.01,100,False)
        for a,b in ((0.02,0.01),(0.04,0.01),(0.08,0.01),(0.08,0.05)):
            r,t=suco(name,ns,K,a,b,1000,True)
            print(f"{name:<10} subspaces={ns:<3} K={K:<4} alpha={a:<5} beta={b:<5} recall {r if r is None else format(r,'.4f')}  {t} ms",flush=True)
