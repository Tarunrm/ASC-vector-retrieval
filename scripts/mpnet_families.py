import numpy as np, os, re, subprocess, time, faiss, hnswlib
from annoy import AnnoyIndex
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
H=os.path.expanduser('~'); W=f'{H}/mpnet'; os.makedirs(W,exist_ok=True); D=768; NQ=500; K=50
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
ds=load_dataset("microsoft/ms_marco","v1.1",split="validation"); pid={}; P=[]; QT=[]
for it in ds:
    ok=False
    for p,s in zip(it['passages']['passage_text'],it['passages']['is_selected']):
        if p not in pid: pid[p]=len(P); P.append(p)
        if s==1: ok=True
    if ok and len(QT)<NQ: QT.append(it['query'])
n=len(P); print("corpus",n,"queries",len(QT),flush=True)
if os.path.exists(f'{W}/X.npy'): X=np.load(f'{W}/X.npy'); Q=np.load(f'{W}/Q.npy')
else:
    m=SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    X=m.encode(P,batch_size=64,normalize_embeddings=True,show_progress_bar=True).astype(np.float32); Q=m.encode(QT,batch_size=64,normalize_embeddings=True).astype(np.float32)
    np.save(f'{W}/X.npy',X); np.save(f'{W}/Q.npy',Q)
faiss.omp_set_num_threads(1); fl=faiss.IndexFlatL2(D); fl.add(X)
G=np.array([fl.search(v[None],K)[1][0] for v in Q])
def run(label,setting,search):
    t=time.time(); L=np.array([search(v) for v in Q]); ms=(time.time()-t)/len(Q)*1000
    rc=np.mean([len(set(L[i].tolist())&set(G[i].tolist()))/K for i in range(len(Q))]); print(f"{label:<14}{setting:<28}{rc:<11.4f}{ms:.3f}",flush=True)
print(f"\n=== MS MARCO, 768-d mpnet embeddings, {n} passages, one thread ===\n{'family':<14}{'setting':<28}{'recall@50':<11}{'ms/query'}")
run('Exact','flat',lambda v:fl.search(v[None],K)[1][0])
h=hnswlib.Index(space='l2',dim=D); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_num_threads(1)
for ef in (50,100,200,400): h.set_ef(ef); run('HNSW',f'ef={ef}',lambda v:h.knn_query(v[None],k=K)[0][0])
del h
qz=faiss.IndexFlatL2(D); pq=faiss.IndexIVFPQ(qz,D,256,D//2,8); pq.train(X); pq.add(X)
for npb in (4,16,64): pq.nprobe=npb; run('IVF-PQ',f'M={D//2} nprobe={npb}',lambda v:pq.search(v[None],K)[1][0])
del pq
for nb in (256,1024):
    lsh=faiss.IndexLSH(D,nb); rf=faiss.IndexRefineFlat(lsh); rf.train(X); rf.add(X)
    for kf in (20,100,500): rf.k_factor=kf; run('LSH+rerank',f'bits={nb} kfactor={kf}',lambda v:rf.search(v[None],K)[1][0])
    del rf,lsh
for nt in (10,50):
    an=AnnoyIndex(D,'euclidean')
    for i in range(n): an.add_item(i,X[i])
    an.build(nt)
    for sk in (1000,5000,25000,100000): run('Annoy',f'trees={nt} search_k={sk}',lambda v:an.get_nns_by_vector(v.tolist(),K,search_k=sk))
    del an
X.tofile(f'{W}/s_base.bin'); Q.tofile(f'{W}/s_query.bin'); G.astype(np.int64).tofile(f'{W}/s_gt.bin')
def suco(a,b,load):
    if os.path.exists(f'{H}/suco_results.bin'): os.remove(f'{H}/suco_results.bin')
    c=['./suco_base','--dataset-path',f'{W}/s_base.bin','--query-path',f'{W}/s_query.bin','--groundtruth-path',f'{W}/s_gt.bin','--dataset-size',str(n+100),'--query-size',str(len(Q)),'--k-size',str(K),'--data-dimensionality',str(D),'--subspace-dimensionality','96','--subspace-num','8','--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{W}/suco_idx']+(['--load-index'] if load else [])
    o=subprocess.run(c,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=7200); o=o.stdout+o.stderr
    r=np.fromfile(f'{H}/suco_results.bin',dtype=np.int32); R=r[2:].reshape(r[0],r[1]); t=float(re.search(r'query time is ([0-9.]+)',o).group(1))
    return np.mean([len(set(R[i].tolist())&set(G[i].tolist()))/K for i in range(len(Q))]),t
suco(0.04,0.01,False)
for a,b in ((0.04,0.01),(0.04,0.05),(0.08,0.01),(0.08,0.05),(0.12,0.05)):
    rc,t=suco(a,b,True); print(f"{'SuCo':<14}{'a='+str(a)+' b='+str(b):<28}{rc:<11.4f}{t:.3f}",flush=True)
