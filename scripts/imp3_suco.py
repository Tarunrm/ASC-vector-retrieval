import numpy as np, os, subprocess, hnswlib
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
n=len(P); print("corpus",n,"queries",len(QT))
X=np.fromfile(f'{W}/base.fvecs',dtype=np.float32).reshape(-1,dim+1)[:,1:].copy()
assert X.shape[0]==n, f"saved embeddings ({X.shape[0]}) do not match the rebuilt corpus ({n})"
Q=np.fromfile(f'{W}/query.fvecs',dtype=np.float32).reshape(-1,dim+1)[:,1:].copy()[:NQ]
GT=np.fromfile(f'{W}/gt.ivecs',dtype=np.int32).reshape(NQ,101)[:,1:]
X.tofile(f'{W}/s_base.bin'); Q.tofile(f'{W}/s_query.bin'); GT.astype(np.int64).tofile(f'{W}/s_gt.bin')
def suco(binary,build):
    c=[f'./{binary}','--dataset-path',f'{W}/s_base.bin','--query-path',f'{W}/s_query.bin','--groundtruth-path',f'{W}/s_gt.bin','--dataset-size',str(n+100),
       '--query-size',str(NQ),'--k-size','100','--data-dimensionality',str(dim),'--subspace-dimensionality','48','--subspace-num','8','--candidate-ratio','0.01',
       '--collision-ratio','0.08','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{W}/suco_idx']+([] if build else ['--load-index'])
    if os.path.exists(f'{H}/suco_results.bin'): os.remove(f'{H}/suco_results.bin')
    subprocess.run(c,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=3600)
    a=np.fromfile(f'{H}/suco_results.bin',dtype=np.int32); return a[2:].reshape(a[0],a[1])
R={}
R['Exact search']=GT
h=hnswlib.Index(space='l2',dim=dim); h.init_index(max_elements=n,ef_construction=200,M=16); h.add_items(X); h.set_ef(200); R['HNSW ef=200']=h.knn_query(Q,k=100)[0]
for nm,f in (('TaCo','taco.bin'),('ASC (Imp1+2, TaCo)','asc.bin')):
    a=np.fromfile(f'{W}/{f}',dtype=np.int32); R[nm]=a[2:].reshape(a[0],a[1])
suco('suco_base',True); R['SuCo']=suco('suco_base',False); R['SuCo + Imp1']=suco('suco_imp1',False)
ce=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); cache={}
def score(qi,ids):
    need=[(qi,int(d)) for d in ids if (qi,int(d)) not in cache]
    if need:
        s=ce.predict([(QT[a],P[b]) for a,b in need],batch_size=64)
        for k_,v in zip(need,s): cache[k_]=v
    return np.array([cache[(qi,int(d))] for d in ids])
def ndcg(r,rel,k=10):
    d=sum(1/np.log2(i+2) for i,x in enumerate(r[:k]) if int(x) in rel); return d/sum(1/np.log2(i+2) for i in range(min(len(rel),k)))
def mrr(r,rel):
    for i,x in enumerate(r[:10]):
        if int(x) in rel: return 1/(i+1)
    return 0.0
per={}; rng=np.random.default_rng(0)
print(f"{'First stage':<22}{'Recall@100':<12}{'nDCG geo':<10}{'nDCG +CE':<10}{'MRR +CE':<9}{'d nDCG+CE vs exact [95% CI]'}")
for name,res in R.items():
    rc=[];ng=[];nr=[];mr=[]
    for i in range(NQ):
        ids=res[i][:100]; rel=REL[i]
        rc.append(len(rel&set(int(x) for x in ids))/len(rel)); ng.append(ndcg(ids,rel))
        rr=ids[np.argsort(-score(i,ids))]; nr.append(ndcg(rr,rel)); mr.append(mrr(rr,rel))
    per[name]=np.array(nr)
    if name=='Exact search': ci='-'
    else:
        d=per[name]-per['Exact search']; bs=[d[rng.integers(0,NQ,NQ)].mean() for _ in range(2000)]
        ci=f"{d.mean():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]"
    print(f"{name:<22}{np.mean(rc):<12.4f}{np.mean(ng):<10.4f}{np.mean(nr):<10.4f}{np.mean(mr):<9.4f}{ci}",flush=True)
