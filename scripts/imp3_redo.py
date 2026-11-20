import numpy as np, os, subprocess, shutil, faiss, hnswlib
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
H=os.path.expanduser('~'); W=f'{H}/imp3_val'; os.makedirs(W,exist_ok=True); NQ=500
ds=load_dataset("microsoft/ms_marco","v1.1",split="validation")
pid={}; P=[]; QT=[]; REL=[]
for it in ds:
    sel=[]
    for p,s in zip(it['passages']['passage_text'],it['passages']['is_selected']):
        if p not in pid: pid[p]=len(P); P.append(p)
        if s==1: sel.append(pid[p])
    if sel and len(QT)<NQ: QT.append(it['query']); REL.append(set(sel))
print("corpus",len(P),"queries",len(QT))
m=SentenceTransformer('all-MiniLM-L6-v2')
X=m.encode(P,batch_size=256,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
Q=m.encode(QT,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32); n,dim=X.shape
def wf(path,A):
    o=np.empty((len(A),dim+1),np.float32); o[:,0]=np.array([dim],np.int32).view(np.float32)[0]; o[:,1:]=A; o.tofile(path)
wf(f'{W}/base.fvecs',X); wf(f'{W}/query.fvecs',Q)
fi=faiss.IndexFlatL2(dim); fi.add(X); _,GT=fi.search(Q,100)
g=np.empty((NQ,101),np.int32); g[:,0]=100; g[:,1:]=GT; g.tofile(f'{W}/gt.ivecs')
def taco(binary,extra,load,tag):
    cmd=[f'./{binary}','--dataset-path',f'{W}/base.fvecs','--query-path',f'{W}/query.fvecs','--groundtruth-path',f'{W}/gt.ivecs','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','100','--data-dimensionality',str(dim),'--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio','0.01','--collision-ratio','0.08','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{W}/idx']+(['--load-index'] if load else [])+extra
    subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True); shutil.copy(f'{H}/taco_results.bin',f'{W}/{tag}.bin')
    a=np.fromfile(f'{W}/{tag}.bin',dtype=np.int32); return a[2:].reshape(a[0],a[1])
R={}
R['Brute force']=GT
h=hnswlib.Index(space='l2',dim=dim); h.init_index(max_elements=n,ef_construction=200,M=16); h.add_items(X); h.set_ef(200); R['HNSW ef=200']=h.knn_query(Q,k=100)[0]
R['TaCo']=taco('taco_base',[],False,'taco')
R['ASC Imp1+2']=taco('taco_imp1',['--K-c','100','--T-c','10'],True,'asc')
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
print(f"{'First stage':<14}{'Recall@100':<12}{'nDCG@10 geo':<13}{'nDCG@10 +CE':<13}{'MRR@10 +CE'}")
for name,res in R.items():
    rc=[];ng=[];nr=[];mr=[]
    for i in range(NQ):
        ids=res[i][:100]; rel=REL[i]
        rc.append(len(rel&set(int(x) for x in ids))/len(rel)); ng.append(ndcg(ids,rel))
        rr=ids[np.argsort(-score(i,ids))]; nr.append(ndcg(rr,rel)); mr.append(mrr(rr,rel))
    print(f"{name:<14}{np.mean(rc):<12.4f}{np.mean(ng):<13.4f}{np.mean(nr):<13.4f}{np.mean(mr):.4f}",flush=True)
