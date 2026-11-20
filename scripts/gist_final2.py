import os, re, subprocess
H=os.path.expanduser('~'); n,dim=250000,960
def run(binary, a, b, extra, label):
    cmd=[f'./{binary}','--dataset-path',f'{H}/gist_sub/base.fvecs','--query-path',f'{H}/gist_sub/query.fvecs',
         '--groundtruth-path',f'{H}/gist_sub/gt.ivecs','--dataset-size',str(n+100),'--query-size','1000','--k-size','50',
         '--data-dimensionality',str(dim),'--subspace-dimensionality','6','--subspace-num','6',
         '--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2',
         '--index-path',f'{H}/gist_sub/idx_sd6','--load-index']+extra
    try: r=subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env={**os.environ,'ASC_THREADS':'1'},timeout=600)
    except subprocess.TimeoutExpired: print(f"{label:<28} TIMEOUT",flush=True); return
    o=r.stdout+r.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    print(f"{label:<28} recall {m.group(1)}  {t.group(1)} ms" if m and t else f"{label} FAILED",flush=True)
for a,b in [(0.08,0.05),(0.04,0.05)]:
    print(f"--- alpha={a} beta={b} (total budget {int(6*a*250000)} < N, clamp cannot bind) ---")
    for lab,bn,ex in [('TaCo baseline','taco_base',[]),('Imp1 only','taco_imp1',[]),
                      ('Imp2 only','taco_base',['--K-c','100','--T-c','10']),('Imp1+2','taco_imp1',['--K-c','100','--T-c','10'])]:
        run(bn,a,b,ex,lab)
