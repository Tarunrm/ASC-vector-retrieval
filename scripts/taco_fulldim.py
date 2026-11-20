import subprocess, os, re
H=os.path.expanduser('~'); D=f'{H}/deep96_subsets'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','ASC_THREADS':'1'}
CFG=[('deep96_1m',1000000,96,6,16,f'{D}/base_1000000.fvecs',f'{D}/queries.fvecs',f'{D}/gt_1000000.ivecs'),
     ('sift_1m',1000000,128,8,16,f'{H}/sift/sift_base.fvecs',f'{H}/sift/sift_query.fvecs',f'{H}/sift/sift_groundtruth.ivecs')]
for name,n,d,ns,sd,b,q,g in CFG:
    idx=f'{H}/fd_idx_taco_{name}'
    for load in (False,True):
        cmd=['./taco_base','--dataset-path',b,'--query-path',q,'--groundtruth-path',g,'--dataset-size',str(n+100),'--query-size','1000','--k-size','50',
             '--data-dimensionality',str(d),'--subspace-dimensionality',str(sd),'--subspace-num',str(ns),'--candidate-ratio','0.005','--collision-ratio','0.04',
             '--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',idx]+(['--load-index'] if load else [])
        o=subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True,env=ENV,timeout=7200); o=o.stdout+o.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    print(f"{name}: TaCo {ns}x{sd} (no truncation)  recall {m.group(1)}  {t.group(1)} ms" if m and t else f"{name} FAILED: {o[-300:]}",flush=True)
