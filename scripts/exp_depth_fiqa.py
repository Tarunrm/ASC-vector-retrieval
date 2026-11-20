import numpy as np, os, time, hnswlib, faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
NQ=500; DEPTHS=(10,20,30,50,100)
corp=load_dataset("mteb/fiqa","corpus")["corpus"]; qs=load_dataset("mteb/fiqa","queries")["queries"]; qr=load_dataset("mteb/fiqa","default")["test"]
P=[(t+" "+x).strip() for t,x in zip(corp["title"],corp["text"])]; cid={c:i for i,c in enumerate(corp["_id"])}
qtext=dict(zip(qs["_id"],qs["text"])); rels={}
for q,c,s in zip(qr["query-id"],qr["corpus-id"],qr["score"]):
    if s>0 and c in cid: rels.setdefault(q,set()).add(cid[c])
qids=[q for q in dict.fromkeys(qr["query-id"]) if q in rels][:NQ]
QT=[qtext[q] for q in qids]; REL=[rels[q] for q in qids]; n=len(P); dim=384
print("passages",n,"queries",len(QT),flush=True)
enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
X=enc.encode(P,batch_size=128,show_progress_bar=False,convert_to_numpy=True).astype(np.float32)
Q=enc.encode(QT,batch_size=128,convert_to_numpy=True).astype(np.float32)
faiss.omp_set_num_threads(1); fl=faiss.IndexFlatL2(dim); fl.add(X)
ex=np.array([fl.search(v[None],100)[1][0] for v in Q])
h=hnswlib.Index(space='l2',dim=dim); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(8); h.add_items(X); h.set_ef(200); h.set_num_threads(1)
hn=np.array([h.knn_query(v[None],k=100)[0][0] for v in Q])
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
def mrr(r,rel,k=10):
    for i,x in enumerate(r[:k]):
        if int(x) in rel: return 1/(i+1)
    return 0.0
print("(check: the FiQA run of the thesis had Recall@100 0.7134 and nDCG@10 before 0.3788 for exact search)")
print(f"{'First stage':<14}{'Depth':<7}{'Recall@D':<10}{'nDCG@10 before':<16}{'nDCG@10 after':<15}{'MRR@10 after':<14}{'Hit@1':<8}{'Hit@5':<8}{'CE ms/query'}")
for name,res in (('Exact search',ex),('HNSW ef=200',hn)):
    for D in DEPTHS:
        rc=[];nb=[];na=[];mr=[];h1=[];h5=[]
        for i in range(len(QT)):
            ids=res[i][:D]; rel=REL[i]; rr=ids[np.argsort(-score(i,ids))]
            rc.append(len(rel&set(int(x) for x in ids))/len(rel)); nb.append(ndcg(ids,rel)); na.append(ndcg(rr,rel)); mr.append(mrr(rr,rel)); h1.append(hit(rr,rel,1)); h5.append(hit(rr,rel,5))
        t=time.time()
        for i in range(30): ce.predict([(QT[i],P[int(d)]) for d in res[i][:D]],batch_size=100)
        ms=(time.time()-t)/30*1000
        print(f"{name:<14}{D:<7}{np.mean(rc):<10.4f}{np.mean(nb):<16.4f}{np.mean(na):<15.4f}{np.mean(mr):<14.4f}{np.mean(h1):<8.3f}{np.mean(h5):<8.3f}{ms:.1f}",flush=True)
