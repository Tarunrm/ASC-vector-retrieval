import numpy as np, os, re, subprocess, struct, time, faiss, hnswlib, resource
H=os.path.expanduser('~'); d=960; os.makedirs(f'{H}/gist_scale',exist_ok=True)
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
qraw=np.fromfile(f'{H}/gist/gist_query.fvecs',dtype=np.float32).reshape(-1,d+1)[:1000,1:].copy()
def suco(n,a,b,idx,nq,load):
    S=f'{H}/gist_scale/{n}'
    c=['/usr/bin/time','-v','./suco','--dataset-path',f'{S}/base.bin','--query-path',f'{S}/query.bin','--groundtruth-path',f'{S}/gt.bin',
       '--dataset-size',str(n+100),'--query-size',str(nq),'--k-size','50','--data-dimensionality',str(d),'--subspace-dimensionality','120','--subspace-num','8',
       '--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',idx]+(['--load-index'] if load else [])
    try: r=subprocess.run(c,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=5400)
    except subprocess.TimeoutExpired: return None
    o=r.stdout+r.stderr
    g=lambda p:(re.search(p,o) or [None,None])[1]
    return dict(rec=g(r'k = 50, \(recall, ratio\) = \(([0-9.]+)'),qt=g(r'average query time is ([0-9.]+)'),rss=g(r'Maximum resident set size \(kbytes\): (\d+)'),w=g(r'Elapsed.*: ([0-9:.]+)'),tail=o[-250:])
for n in [500000,1000000]:
    print(f"\n=== Gist-{n} ===",flush=True); S=f'{H}/gist_scale/{n}'; os.makedirs(S,exist_ok=True)
    X=np.fromfile(f'{H}/gist/gist_base.fvecs',dtype=np.float32,count=n*(d+1)).reshape(n,d+1)[:,1:].copy()
    fi=faiss.IndexFlatL2(d); fi.add(X); _,G=fi.search(qraw,50); del fi
    X.tofile(f'{S}/base.bin'); qraw.tofile(f'{S}/query.bin'); G.astype(np.int64).tofile(f'{S}/gt.bin')
    rec=lambda L:np.mean([len(set(L[i].tolist())&set(G[i].tolist()))/50 for i in range(len(L))])
    idx=f'{H}/gist_scale/suco_idx_{n}'
    b=suco(n,0.04,0.01,idx,100,False)
    print("SuCo build:", f"wall {b['w']}  peak {int(b['rss'])/1e6:.2f} GB" if b and b['rss'] else f"FAILED {b['tail'] if b else 'timeout'}",flush=True)
    if b and b['rss']:
        for a_,b_ in [(0.04,0.01),(0.04,0.05),(0.08,0.05),(0.12,0.05)]:
            r=suco(n,a_,b_,idx,1000,True)
            print(f"SuCo alpha={a_} beta={b_}: recall {r['rec']}  {r['qt']} ms" if r and r['rec'] else f"SuCo {a_} {b_} FAILED",flush=True)
    t=time.time(); h=hnswlib.Index(space='l2',dim=d); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X)
    print(f"HNSW build (8 threads): {time.time()-t:.0f}s  python peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB",flush=True)
    h.set_num_threads(1)
    for ef in [100,200,400,800]:
        h.set_ef(ef); L=[]; t=time.time()
        for v in qraw: L.append(h.knn_query(v[None],k=50)[0][0])
        print(f"HNSW ef={ef}: recall {rec(np.array(L)):.4f}  {(time.time()-t)/1000*1000:.3f} ms",flush=True)
    del h, X
