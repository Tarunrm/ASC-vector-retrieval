import numpy as np, os, re, subprocess, time, hnswlib, faiss
from datasets import load_dataset
from sentence_transformers import CrossEncoder
H=os.path.expanduser('~'); W=f'{H}/imp3_val'; NQ=500; dim=384
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
ds=load_dataset("microsoft/ms_marco","v1.1",split="validation")
pid={}; P=[]; QT=[]; REL=[]
for it in ds:
    sel=[]
    for p,s in zip(it['passages']['passage_text'],it['passages']['is_selected']):
        if p not in pid: pid[p]=len(P); P.append(p)
        if s==1: sel.append(pid[p])
    if sel and len(QT)<NQ: QT.append(it['query']); REL.append(set(sel))
n=len(P); X=np.fromfile(f'{W}/base.fvecs',dtype=np.float32).reshape(-1,dim+1)[:,1:].copy()
assert X.shape[0]==n, "saved embeddings do not match the rebuilt corpus"
Q=np.fromfile(f'{W}/query.fvecs',dtype=np.float32).reshape(-1,dim+1)[:,1:].copy()[:NQ]
GT=np.fromfile(f'{W}/gt.ivecs',dtype=np.int32).reshape(NQ,101)[:,1:]
def suco(binary):
    if os.path.exists(f'{H}/suco_results.bin'): os.remove(f'{H}/suco_results.bin')
    c=[f'./{binary}','--dataset-path',f'{W}/s_base.bin','--query-path',f'{W}/s_query.bin','--groundtruth-path',f'{W}/s_gt.bin','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','100','--data-dimensionality',str(dim),'--subspace-dimensionality','48','--subspace-num','8','--candidate-ratio','0.01','--collision-ratio','0.08','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{W}/suco_idx','--load-index']
    o=subprocess.run(c,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=3600); o=o.stdout+o.stderr
    a=np.fromfile(f'{H}/suco_results.bin',dtype=np.int32); return a[2:].reshape(a[0],a[1]), float(re.search(r'query time is ([0-9.]+)',o).group(1))
R={}; LAT={}
faiss.omp_set_num_threads(1); fl=faiss.IndexFlatL2(dim); fl.add(X)
t=time.time(); ex=np.array([fl.search(v[None],100)[1][0] for v in Q]); LAT['Exact search']=(time.time()-t)/NQ*1000; R['Exact search']=ex
h=hnswlib.Index(space='l2',dim=dim); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_ef(200); h.set_num_threads(1)
t=time.time(); hn=np.array([h.knn_query(v[None],k=100)[0][0] for v in Q]); LAT['HNSW (ef=200)']=(time.time()-t)/NQ*1000; R['HNSW (ef=200)']=hn
R['SuCo'],LAT['SuCo']=suco('suco_base')
for nm,f in (('TaCo','taco.bin'),('ASC (Imp1+2, TaCo)','asc.bin')):
    a=np.fromfile(f'{W}/{f}',dtype=np.int32); R[nm]=a[2:].reshape(a[0],a[1])
ce=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); cache={}
def score(qi,ids):
    need=[(qi,int(d)) for d in ids if (qi,int(d)) not in cache]
    if need:
        s=ce.predict([(QT[a],P[b]) for a,b in need],batch_size=100)
        for k_,v in zip(need,s): cache[k_]=v
    return np.array([cache[(qi,int(d))] for d in ids])
def hit(r,rel,k): return float(any(int(x) in rel for x in r[:k]))
def ndcg(r,rel,k=10):
    d=sum(1/np.log2(i+2) for i,x in enumerate(r[:k]) if int(x) in rel); return d/sum(1/np.log2(i+2) for i in range(min(len(rel),k)))
print(f"{'First stage':<22}{'Rec@100':<9}{'Hit@5 pre':<10}{'Hit@1':<7}{'Hit@5':<7}{'Hit@10':<8}{'nDCG@10':<9}{'1st-stage ms'}")
for name,res in R.items():
    h1=[];h5=[];h10=[];h5p=[];nd=[];rc=[]
    for i in range(NQ):
        ids=res[i][:100]; rel=REL[i]; rr=ids[np.argsort(-score(i,ids))]
        rc.append(len(rel&set(int(x) for x in ids))/len(rel)); h5p.append(hit(ids,rel,5))
        h1.append(hit(rr,rel,1)); h5.append(hit(rr,rel,5)); h10.append(hit(rr,rel,10)); nd.append(ndcg(rr,rel))
    lat=f"{LAT[name]:.3f}" if name in LAT else "-"
    print(f"{name:<22}{np.mean(rc):<9.4f}{np.mean(h5p):<10.4f}{np.mean(h1):<7.4f}{np.mean(h5):<7.4f}{np.mean(h10):<8.4f}{np.mean(nd):<9.4f}{lat}",flush=True)
# cross-encoder cost for 100 candidates per query, measured on 30 queries, no cache
t=time.time()
for i in range(30): ce.predict([(QT[i],P[int(d)]) for d in hn[i][:100]],batch_size=100)
print(f"\nCross-encoder re-ranking of 100 candidates: {(time.time()-t)/30*1000:.1f} ms per query (default CPU threads)")
