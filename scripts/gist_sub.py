import numpy as np, os, faiss, subprocess, re, time, struct, hnswlib
H=os.path.expanduser('~'); os.makedirs(f'{H}/gist_sub',exist_ok=True)
n,dim=250000,960; rec_bytes=4+4*dim
raw=open(f'{H}/gist/gist_base.fvecs','rb').read(n*rec_bytes)
open(f'{H}/gist_sub/base.fvecs','wb').write(raw)
X=np.frombuffer(raw,dtype=np.float32).reshape(n,dim+1)[:,1:].copy(); del raw
qraw=open(f'{H}/gist/gist_query.fvecs','rb').read()
open(f'{H}/gist_sub/query.fvecs','wb').write(qraw)
Q=np.frombuffer(qraw,dtype=np.float32).reshape(-1,dim+1)[:1000,1:].copy()
fi=faiss.IndexFlatL2(dim); fi.add(X); _,G=fi.search(Q,100)
with open(f'{H}/gist_sub/gt.ivecs','wb') as f:
    for r in G: f.write(struct.pack('i',100)); f.write(r.astype(np.int32).tobytes())
rec=lambda L:np.mean([len(set(L[i].tolist())&set(G[i][:50].tolist()))/50 for i in range(len(L))])
h=hnswlib.Index(space='l2',dim=dim); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_num_threads(1)
for ef in [50,100,200,400,800]:
    h.set_ef(ef); L=[]; t=time.time()
    for v in Q: L.append(h.knn_query(v[None],k=50)[0][0])
    print(f"HNSW ef={ef}: recall {rec(np.array(L)):.4f}, {(time.time()-t)/len(Q)*1000:.3f} ms",flush=True)
idx=f'{H}/gist_sub/idx'
for i,a in enumerate(['0.04','0.08','0.12']):
    cmd=['./taco_base','--dataset-path',f'{H}/gist_sub/base.fvecs','--query-path',f'{H}/gist_sub/query.fvecs','--groundtruth-path',f'{H}/gist_sub/gt.ivecs','--dataset-size',str(n+100),'--query-size','1000','--k-size','50','--data-dimensionality',str(dim),'--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio','0.01','--collision-ratio',a,'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',idx]+(['--load-index'] if i else [])
    o=subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env={**os.environ,'ASC_THREADS':'1'}); o=o.stdout+o.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    print(f"TaCo alpha={a}: recall {m.group(1)}, {t.group(1)} ms" if m and t else f"TaCo alpha={a} FAILED\n{o[-400:]}",flush=True)
