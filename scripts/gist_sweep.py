import os, re, subprocess
H=os.path.expanduser('~'); n,dim=250000,960
print("subdim alpha beta  recall  ms/query")
for sd in [6,12,16]:
    idx=f'{H}/gist_sub/idx_sd{sd}'
    first=True
    for a in ['0.12','0.2','0.3']:
        cmd=['./taco_base','--dataset-path',f'{H}/gist_sub/base.fvecs','--query-path',f'{H}/gist_sub/query.fvecs',
             '--groundtruth-path',f'{H}/gist_sub/gt.ivecs','--dataset-size',str(n+100),'--query-size','1000','--k-size','50',
             '--data-dimensionality',str(dim),'--subspace-dimensionality',str(sd),'--subspace-num','6',
             '--candidate-ratio','0.01','--collision-ratio',a,'--kmeans-num-centroid','50','--kmeans-num-iters','2',
             '--index-path',idx]+([] if first else ['--load-index'])
        first=False
        o=subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env={**os.environ,'ASC_THREADS':'1'}); o=o.stdout+o.stderr
        m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
        print(f"{sd:<6} {a:<5} 0.01  {m.group(1)}  {t.group(1)}" if m and t else f"{sd} {a} FAILED: {o[-300:]}",flush=True)
