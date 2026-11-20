import os, re, subprocess
H=os.path.expanduser('~'); n,dim=250000,960
def run(binary, idx, a, b, extra, label):
    cmd=[f'./{binary}','--dataset-path',f'{H}/gist_sub/base.fvecs','--query-path',f'{H}/gist_sub/query.fvecs',
         '--groundtruth-path',f'{H}/gist_sub/gt.ivecs','--dataset-size',str(n+100),'--query-size','1000','--k-size','50',
         '--data-dimensionality',str(dim),'--subspace-dimensionality','6','--subspace-num','6',
         '--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2',
         '--index-path',idx,'--load-index']+extra
    try:
        r=subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env={**os.environ,'ASC_THREADS':'1'},timeout=600)
    except subprocess.TimeoutExpired:
        print(f"{label:<34} TIMEOUT (>600s)",flush=True); return
    o=r.stdout+r.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    print(f"{label:<34} recall {m.group(1)}  {t.group(1)} ms" if m and t else f"{label} FAILED: {o[-200:]}",flush=True)
idx=f'{H}/gist_sub/idx_sd6'
print("--- A. Ablation on Gist, alpha=0.2 beta=0.01 ---")
for lab,bn,ex in [('TaCo baseline','taco_base',[]),('Imp1 only','taco_imp1',[]),
                  ('Imp2 only','taco_base',['--K-c','100','--T-c','10']),
                  ('Imp1+2','taco_imp1',['--K-c','100','--T-c','10'])]:
    run(bn,idx,0.2,0.01,ex,lab)
print("--- B. beta sweep, alpha=0.2 ---")
for b in [0.01,0.02,0.05,0.1]:
    run('taco_base',idx,0.2,b,[],f'beta={b}')
