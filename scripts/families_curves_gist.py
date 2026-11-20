import numpy as np, os, time, faiss, hnswlib
from annoy import AnnoyIndex
H=os.path.expanduser('~'); D=f'{H}/deep96_subsets'; faiss.omp_set_num_threads(1)
CFG=[('Gist-250K',f'{H}/gist_sub/base.fvecs',f'{H}/gist_sub/query.fvecs',f'{H}/gist_sub/gt.ivecs',250000,960)]
def fv(p,n,d): return np.ascontiguousarray(np.fromfile(p,dtype=np.float32,count=n*(d+1)).reshape(n,d+1)[:,1:])
def run(Q,G,search,label,setting):
    t=time.time(); L=np.array([search(v) for v in Q]); ms=(time.time()-t)/len(Q)*1000
    rc=np.mean([len(set(L[i].tolist())&set(G[i].tolist()))/50 for i in range(len(L))])
    print(f"{label:<14}{setting:<26}{rc:<11.4f}{ms:.3f}",flush=True)
for name,b,q,g,n,d in CFG:
    X=fv(b,n,d); Q=fv(q,1000,d); G=np.fromfile(g,dtype=np.int32).reshape(-1,101)[:1000,1:51]
    print(f"\n=== {name}, one thread, one query at a time ===\n{'family':<14}{'setting':<26}{'recall@50':<11}{'ms/query'}")
    fl=faiss.IndexFlatL2(d); fl.add(X); run(Q,G,lambda v:fl.search(v[None],50)[1][0],'Exact','flat'); del fl
    h=hnswlib.Index(space='l2',dim=d); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_num_threads(1)
    for ef in (50,100,200,400):
        h.set_ef(ef); run(Q,G,lambda v:h.knn_query(v[None],k=50)[0][0],'HNSW',f'ef={ef}')
    del h
    qz=faiss.IndexFlatL2(d); pq=faiss.IndexIVFPQ(qz,d,1024,d//2,8); pq.train(X[:300000]); pq.add(X)
    for npb in (4,16,64):
        pq.nprobe=npb; run(Q,G,lambda v:pq.search(v[None],50)[1][0],'IVF-PQ',f'M={d//2} nprobe={npb}')
    del pq
    for nb in (256,1024):
        lsh=faiss.IndexLSH(d,nb); rf=faiss.IndexRefineFlat(lsh); rf.train(X); rf.add(X)
        for kf in (20,100,500):
            rf.k_factor=kf; run(Q,G,lambda v:rf.search(v[None],50)[1][0],'LSH+rerank',f'bits={nb} kfactor={kf}')
        del rf,lsh
    for nt in (10,50):
        an=AnnoyIndex(d,'euclidean')
        for i in range(n): an.add_item(i,X[i])
        an.build(nt)
        for sk in (1000,5000,25000,100000):
            run(Q,G,lambda v:an.get_nns_by_vector(v.tolist(),50,search_k=sk),'Annoy',f'trees={nt} search_k={sk}')
        del an
