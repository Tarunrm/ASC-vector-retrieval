import numpy as np, os, re, subprocess, time, hnswlib
H=os.path.expanduser('~'); D=f'{H}/deep96_subsets'; S=f'{H}/suco_data/deep96_1m'; n,d,R=1000000,96,5
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','ASC_THREADS':'1'}
def run(cmd,cwd):
    o=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,env=ENV,timeout=7200); o=o.stdout+o.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    return (float(m.group(1)),float(t.group(1))) if m and t else (None,None)
res={'SuCo':[],'TaCo':[]}
for r in range(R):
    # no --load-index: each run rebuilds the index from scratch, then answers the 1000 queries
    res['SuCo'].append(run(['./suco_base','--dataset-path',f'{S}/base.bin','--query-path',f'{S}/query.bin','--groundtruth-path',f'{S}/gt.bin','--dataset-size',str(n+100),'--query-size','1000','--k-size','50','--data-dimensionality','96','--subspace-dimensionality','16','--subspace-num','6','--candidate-ratio','0.01','--collision-ratio','0.04','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/rb_suco_{r}'],f'{H}/SuCo'))
    res['TaCo'].append(run(['./taco_base','--dataset-path',f'{D}/base_1000000.fvecs','--query-path',f'{D}/queries.fvecs','--groundtruth-path',f'{D}/gt_1000000.ivecs','--dataset-size',str(n+100),'--query-size','1000','--k-size','50','--data-dimensionality','96','--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio','0.01','--collision-ratio','0.04','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/rb_taco_{r}'],f'{H}/TaCo'))
    print(f"build {r+1}/{R}: SuCo {res['SuCo'][-1]}  TaCo {res['TaCo'][-1]}",flush=True)
X=np.fromfile(f'{D}/base_1000000.fvecs',dtype=np.float32).reshape(n,d+1)[:,1:].copy()
Q=np.fromfile(f'{D}/queries.fvecs',dtype=np.float32,count=1000*(d+1)).reshape(1000,d+1)[:,1:].copy()
G=np.fromfile(f'{D}/gt_1000000.ivecs',dtype=np.int32).reshape(-1,101)[:1000,1:51]
for ef in (100,200): res[f'HNSW ef={ef}']=[]
for r in range(R):
    h=hnswlib.Index(space='l2',dim=d); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_num_threads(1)
    for ef in (100,200):
        h.set_ef(ef); L=[]; t=time.time()
        for v in Q: L.append(h.knn_query(v[None],k=50)[0][0])
        rc=np.mean([len(set(L[i].tolist())&set(G[i].tolist()))/50 for i in range(1000)]); res[f'HNSW ef={ef}'].append((rc,(time.time()-t)/1000*1000))
    print(f"HNSW build {r+1}/{R} done",flush=True); del h
print(f"\nDeep96-1M, {R} independent builds each, one thread, 1000 queries")
print(f"{'system':<14}{'recall mean':<13}{'recall sd':<11}{'recall min-max':<18}{'ms mean':<9}{'ms sd'}")
for k,v in res.items():
    v=[x for x in v if x[0] is not None]; r=np.array([x[0] for x in v]); t=np.array([x[1] for x in v])
    print(f"{k:<14}{r.mean():<13.4f}{r.std(ddof=1):<11.4f}{r.min():.4f}-{r.max():.4f}   {t.mean():<9.3f}{t.std(ddof=1):.3f}")
