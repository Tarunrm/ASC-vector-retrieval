import numpy as np, os, re, subprocess, time, hnswlib, faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
H=os.path.expanduser('~'); W=f'{H}/fiqa'; os.makedirs(W,exist_ok=True); D=384; NQMAX=500
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
L=load_dataset
try:
    cor=L("mteb/fiqa","corpus",split="corpus"); qs=L("mteb/fiqa","queries",split="queries"); qr=L("mteb/fiqa",split="test")
except Exception as e:
    print("mteb/fiqa failed, trying BeIR:",repr(e)[:300],flush=True)
    cor=L("BeIR/fiqa","corpus",split="corpus"); qs=L("BeIR/fiqa","queries",split="queries"); qr=L("BeIR/fiqa-qrels",split="test")
P=[(t or '')+' '+(x or '') for t,x in zip(cor['title'],cor['text'])]; ids=[str(i) for i in cor['_id']]; pid={i:k for k,i in enumerate(ids)}
qtext={str(i):t for i,t in zip(qs['_id'],qs['text'])}; rel={}
for a,b,s in zip(qr['query-id'],qr['corpus-id'],qr['score']):
    if s>0 and str(b) in pid and str(a) in qtext: rel.setdefault(str(a),set()).add(pid[str(b)])
qids=list(rel)[:NQMAX]; NQ=len(qids); QT=[qtext[q] for q in qids]; REL=[rel[q] for q in qids]; n=len(P)
print("corpus",n,"queries",NQ,flush=True)
if os.path.exists(f'{W}/X.npy') and np.load(f'{W}/Q.npy').shape[0]==NQ: X=np.load(f'{W}/X.npy'); Q=np.load(f'{W}/Q.npy')
else:
    m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    X=m.encode(P,batch_size=128,normalize_embeddings=True,show_progress_bar=True).astype(np.float32)
    Q=m.encode(QT,batch_size=128,normalize_embeddings=True).astype(np.float32); np.save(f'{W}/X.npy',X); np.save(f'{W}/Q.npy',Q)
fl=faiss.IndexFlatL2(D); fl.add(X); faiss.omp_set_num_threads(1)
t=time.time(); G=np.array([fl.search(v[None],100)[1][0] for v in Q]); T={'Exact search':(time.time()-t)/NQ*1000}
hd=lambda m,d: np.full((m,1),d,dtype=np.int32)
np.hstack([hd(n,D).view(np.float32),X]).astype(np.float32).tofile(f'{W}/base.fvecs')
np.hstack([hd(NQ,D).view(np.float32),Q]).astype(np.float32).tofile(f'{W}/query.fvecs')
np.hstack([hd(NQ,100),G.astype(np.int32)]).tofile(f'{W}/gt.ivecs')
X.tofile(f'{W}/s_base.bin'); Q.tofile(f'{W}/s_query.bin'); G.astype(np.int64).tofile(f'{W}/s_gt.bin')
def run(cmd,cwd,dump):
    if os.path.exists(f'{H}/{dump}'): os.remove(f'{H}/{dump}')
    r=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,env=ENV,timeout=7200); o=r.stdout+r.stderr
    t=re.search(r'query time is ([0-9.]+)',o); a=np.fromfile(f'{H}/{dump}',dtype=np.int32)
    return a[2:].reshape(a[0],a[1]),(float(t.group(1)) if t else None)
R={'Exact search':G}
h=hnswlib.Index(space='l2',dim=D); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_ef(200); h.set_num_threads(1)
t=time.time(); R['HNSW (ef=200)']=np.array([h.knn_query(v[None],k=100)[0][0] for v in Q]); T['HNSW (ef=200)']=(time.time()-t)/NQ*1000; del h
sc=['./suco_base','--dataset-path',f'{W}/s_base.bin','--query-path',f'{W}/s_query.bin','--groundtruth-path',f'{W}/s_gt.bin','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','100','--data-dimensionality',str(D),'--subspace-dimensionality','48','--subspace-num','8','--candidate-ratio','0.01','--collision-ratio','0.08','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{W}/suco_idx']
run(sc,f'{H}/SuCo','suco_results.bin'); R['SuCo'],T['SuCo']=run(sc+['--load-index'],f'{H}/SuCo','suco_results.bin')
tc=['./taco_base','--dataset-path',f'{W}/base.fvecs','--query-path',f'{W}/query.fvecs','--groundtruth-path',f'{W}/gt.ivecs','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','100','--data-dimensionality',str(D),'--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio','0.01','--collision-ratio','0.08','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{W}/taco_idx']
R['TaCo'],_=run(tc,f'{H}/TaCo','taco_results.bin')
ta=['./taco_imp1']+tc[1:]+['--K-c','100','--T-c','10','--load-index']; R['ASC (Imp1+2, TaCo)'],_=run(ta,f'{H}/TaCo','taco_results.bin')
ce=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); cache={}
def score(qi,idl):
    need=[(qi,int(d)) for d in idl if (qi,int(d)) not in cache]
    if need:
        s=ce.predict([(QT[a],P[b]) for a,b in need],batch_size=64)
        for k_,v in zip(need,s): cache[k_]=v
    return np.array([cache[(qi,int(d))] for d in idl])
def ndcg(r,rl,k=10):
    d=sum(1/np.log2(i+2) for i,x in enumerate(r[:k]) if int(x) in rl); return d/sum(1/np.log2(i+2) for i in range(min(len(rl),k)))
hit=lambda r,rl,k: float(any(int(x) in rl for x in r[:k]))
rng=np.random.default_rng(0); per={}
print(f"\nFiQA, {NQ} queries, corpus {n}\n{'First stage':<22}{'Rec@100':<9}{'nDCG pre':<9}{'nDCG +CE':<9}{'Hit@1':<7}{'Hit@5':<7}{'Hit@10':<8}{'ms':<7}{'d nDCG vs exact [95% CI]'}")
for name,res in R.items():
    rc=[];ng=[];nr=[];h1=[];h5=[];h10=[]
    for i in range(NQ):
        idl=res[i][:100]; rl=REL[i]; rr=idl[np.argsort(-score(i,idl))]
        rc.append(len(rl&set(int(x) for x in idl))/len(rl)); ng.append(ndcg(idl,rl)); nr.append(ndcg(rr,rl)); h1.append(hit(rr,rl,1)); h5.append(hit(rr,rl,5)); h10.append(hit(rr,rl,10))
    per[name]=np.array(nr)
    if name=='Exact search': ci='-'
    else:
        d=per[name]-per['Exact search']; bs=[d[rng.integers(0,NQ,NQ)].mean() for _ in range(2000)]; ci=f"{d.mean()*100:+.2f} [{np.percentile(bs,2.5)*100:+.2f}, {np.percentile(bs,97.5)*100:+.2f}] pts"
    tm=f"{T[name]:.2f}" if name in T and T[name] else "-"
    print(f"{name:<22}{np.mean(rc):<9.4f}{np.mean(ng):<9.4f}{np.mean(nr):<9.4f}{np.mean(h1):<7.3f}{np.mean(h5):<7.3f}{np.mean(h10):<8.3f}{tm:<7}{ci}",flush=True)
