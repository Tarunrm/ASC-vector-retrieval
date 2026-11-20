import numpy as np, os, re, subprocess, time, torch
from datasets import load_dataset
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
H=os.path.expanduser('~'); W=f'{H}/imp3_val'; NQ=100; dim=384; NS,SD,AN=8,48,0.08
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
ds=load_dataset("microsoft/ms_marco","v1.1",split="validation"); pid={}; P=[]; QT=[]; REL=[]
for it in ds:
    sel=[]
    for p,s in zip(it['passages']['passage_text'],it['passages']['is_selected']):
        if p not in pid: pid[p]=len(P); P.append(p)
        if s==1: sel.append(pid[p])
    if sel and len(QT)<NQ: QT.append(it['query']); REL.append(set(sel))
n=len(P); X=np.fromfile(f'{W}/base.fvecs',dtype=np.float32).reshape(-1,dim+1)[:,1:].copy(); assert X.shape[0]==n
Q=np.fromfile(f'{W}/query.fvecs',dtype=np.float32).reshape(-1,dim+1)[:,1:].copy()[:NQ]
if os.path.exists(f'{H}/suco_results.bin'): os.remove(f'{H}/suco_results.bin')
c=['./suco_base','--dataset-path',f'{W}/s_base.bin','--query-path',f'{W}/s_query.bin','--groundtruth-path',f'{W}/s_gt.bin','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','100','--data-dimensionality',str(dim),'--subspace-dimensionality',str(SD),'--subspace-num',str(NS),'--candidate-ratio','0.01','--collision-ratio',str(AN),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{W}/suco_idx','--load-index']
subprocess.run(c,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=3600)
a=np.fromfile(f'{H}/suco_results.bin',dtype=np.int32); cand=a[2:].reshape(a[0],a[1])[:NQ,:100]
# SC-score: true subspace collisions (alpha-n nearest in each subspace), read at the candidates
an=int(AN*n); Xs=[np.ascontiguousarray(X[:,s*SD:(s+1)*SD]) for s in range(NS)]; sq=[(x*x).sum(1) for x in Xs]
SC=np.zeros((NQ,100),np.float32)
for i in range(NQ):
    cnt=np.zeros(n,np.uint8)
    for s in range(NS): cnt[np.argpartition(sq[s]-2*Xs[s]@Q[i,s*SD:(s+1)*SD],an)[:an]]+=1
    SC[i]=cnt[cand[i]]
COS=np.stack([X[cand[i]]@Q[i] for i in range(NQ)])
ce=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
CE=np.stack([ce.predict([(QT[i],P[int(d)]) for d in cand[i]],batch_size=100) for i in range(NQ)])
cf=f'{W}/coh_{NQ}.npy'
if os.path.exists(cf): COH=np.load(cf); lm_s=None
else:
    tok=AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m'); tok.pad_token=tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m').eval(); suffix=tok("\n\nQuestion:",add_special_tokens=False)['input_ids']; COH=np.zeros((NQ,100),np.float32); t0=time.time()
    for i in range(NQ):
        qi=tok(" "+QT[i],add_special_tokens=False)['input_ids'][:24]
        pre=[tok(P[int(d)],add_special_tokens=False)['input_ids'][:150]+suffix for d in cand[i]]
        for b in range(0,100,25):
            seqs=[p+qi for p in pre[b:b+25]]; Lm=max(map(len,seqs)); ids=torch.full((len(seqs),Lm),tok.pad_token_id); att=torch.zeros_like(ids)
            for j,s in enumerate(seqs): ids[j,:len(s)]=torch.tensor(s); att[j,:len(s)]=1
            with torch.no_grad():
                hs=lm.transformer(input_ids=ids,attention_mask=att).last_hidden_state
                for j,p in enumerate(pre[b:b+25]):
                    st=len(p); lp=torch.log_softmax(lm.lm_head(hs[j,st-1:st-1+len(qi)]).float(),-1); COH[i,b+j]=lp[torch.arange(len(qi)),torch.tensor(qi)].mean().item()
        if (i+1)%10==0: print(f"coherence {i+1}/{NQ}  {(time.time()-t0)/(i+1):.1f} s per query",flush=True)
    np.save(cf,COH); lm_s=(time.time()-t0)/NQ
zs=lambda m:(m-m.mean(1,keepdims=True))/(m.std(1,keepdims=True)+1e-9); Z={'sc':zs(SC),'cos':zs(COS),'coh':zs(COH),'ce':zs(CE)}
def ndcg(r,rl,k=10):
    d=sum(1/np.log2(i+2) for i,x in enumerate(r[:k]) if x in rl); return d/sum(1/np.log2(i+2) for i in range(min(len(rl),k)))
def met(S):
    nd=np.zeros(NQ);mr=np.zeros(NQ);h1=np.zeros(NQ);h5=np.zeros(NQ)
    for i in range(NQ):
        r=[int(cand[i][j]) for j in np.argsort(-S[i],kind='stable')]; rl=REL[i]; nd[i]=ndcg(r,rl)
        mr[i]=next((1/(k+1) for k,x in enumerate(r[:10]) if x in rl),0.0); h1[i]=float(r[0] in rl); h5[i]=float(any(x in rl for x in r[:5]))
    return dict(nd=nd,mr=mr,h1=h1,h5=h5)
def grid(k,step=10):
    out=[]
    def rec(pre,left,kk):
        if kk==1: out.append(pre+[left]); return
        for v in range(left+1): rec(pre+[v],left-v,kk-1)
    rec([],step,k); return np.array(out)/step
def cv(keys):
    Wg=grid(len(keys)); allm=[met(sum(w*Z[k] for w,k in zip(wt,keys))) for wt in Wg]; ND=np.stack([m['nd'] for m in allm]); out={x:np.zeros(NQ) for x in ('nd','mr','h1','h5')}; chosen=[]
    for te in np.array_split(np.arange(NQ),5):
        tr=np.setdiff1d(np.arange(NQ),te); b=int(np.argmax(ND[:,tr].mean(1))); chosen.append(Wg[b])
        for x in out: out[x][te]=allm[b][x][te]
    return out,chosen
M={'Cosine (geometric order)':met(Z['cos']),'SC-score only':met(Z['sc']),'Coherence only':met(Z['coh']),'Cross-encoder only':met(Z['ce'])}
M['Composite: SC + cosine + coherence (CV)'],w3=cv(['sc','cos','coh']); M['Composite + cross-encoder (CV)'],w4=cv(['sc','cos','coh','ce'])
rng=np.random.default_rng(0)
def ci(d): bs=[d[rng.integers(0,NQ,NQ)].mean() for _ in range(2000)]; return f"{d.mean()*100:+.2f} [{np.percentile(bs,2.5)*100:+.2f}, {np.percentile(bs,97.5)*100:+.2f}]"
print(f"\n{NQ} MS MARCO queries, SuCo top-100 candidates\n{'Ranking':<42}{'nDCG@10':<9}{'MRR@10':<8}{'Hit@1':<7}{'Hit@5':<7}{'vs cosine, pts [95% CI]':<30}{'vs cross-encoder, pts [95% CI]'}")
for k,m in M.items(): print(f"{k:<42}{m['nd'].mean():<9.4f}{m['mr'].mean():<8.4f}{m['h1'].mean():<7.3f}{m['h5'].mean():<7.3f}{ci(m['nd']-M['Cosine (geometric order)']['nd']):<30}{ci(m['nd']-M['Cross-encoder only']['nd'])}")
print("\nTuned weights per fold (SC, cosine, coherence):",[list(w) for w in w3]); print("Tuned weights per fold (SC, cosine, coherence, cross-encoder):",[list(w) for w in w4])
if lm_s: print(f"\nCoherence scoring: {lm_s:.1f} s per query for 100 candidates on this CPU")
