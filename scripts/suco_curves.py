import numpy as np, os, re, subprocess, time, hnswlib
H=os.path.expanduser('~'); S=f'{H}/suco_data'; D=f'{H}/deep96_subsets'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
CFG=[('deep96_2m',2000000,96,6,16,f'{D}/base_2000000.fvecs',f'{D}/queries.fvecs',f'{D}/gt_2000000.ivecs'),
     ('sift_1m',1000000,128,8,16,f'{H}/sift/sift_base.fvecs',f'{H}/sift/sift_query.fvecs',f'{H}/sift/sift_groundtruth.ivecs')]
GRID=[(a,b) for a in (0.02,0.04,0.08,0.12) for b in (0.005,0.01,0.02,0.05)]
def suco(name,n,d,ns,sd,a,b,idx,load):
    c=['./suco_base','--dataset-path',f'{S}/{name}/base.bin','--query-path',f'{S}/{name}/query.bin','--groundtruth-path',f'{S}/{name}/gt.bin',
       '--dataset-size',str(n+100),'--query-size','1000','--k-size','50','--data-dimensionality',str(d),'--subspace-dimensionality',str(sd),
       '--subspace-num',str(ns),'--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',idx]+(['--load-index'] if load else [])
    try: r=subprocess.run(c,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=3600)
    except subprocess.TimeoutExpired: return None,None,'timeout'
    o=r.stdout+r.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    return (float(m.group(1)),float(t.group(1)),'') if m and t else (None,None,o[-200:])
for name,n,d,ns,sd,base,q,g in CFG:
    print(f"\n=== {name}, one thread ===",flush=True)
    idx=f'{H}/sv_idx_suco_{name}'
    if not os.path.exists(idx):
        print("building SuCo index (not found)...",flush=True); suco(name,n,d,ns,sd,0.04,0.005,idx,False)
    print(f"{'method':<8}{'setting':<22}{'recall@50':<11}{'ms/query'}")
    for a,b in GRID:
        r,t,err=suco(name,n,d,ns,sd,a,b,idx,True)
        print(f"{'SuCo':<8}{'a='+str(a)+' b='+str(b):<22}{r:<11.4f}{t:.3f}" if r is not None else f"SuCo a={a} b={b} FAILED {err}",flush=True)
    X=np.fromfile(base,dtype=np.float32,count=n*(d+1)).reshape(n,d+1)[:,1:].copy()
    Q=np.fromfile(q,dtype=np.float32,count=1000*(d+1)).reshape(1000,d+1)[:,1:].copy()
    G=np.fromfile(g,dtype=np.int32).reshape(-1,101)[:1000,1:51]
    rec=lambda L:np.mean([len(set(L[i].tolist())&set(G[i].tolist()))/50 for i in range(len(L))])
    t0=time.time(); h=hnswlib.Index(space='l2',dim=d); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X)
    print(f"HNSW build (8 threads): {time.time()-t0:.0f}s",flush=True); h.set_num_threads(1)
    for ef in (50,100,200,400,800):
        h.set_ef(ef); L=[]; t0=time.time()
        for v in Q: L.append(h.knn_query(v[None],k=50)[0][0])
        print(f"{'HNSW':<8}{'ef='+str(ef):<22}{rec(np.array(L)):<11.4f}{(time.time()-t0)/len(Q)*1000:.3f}",flush=True)
    del h,X
