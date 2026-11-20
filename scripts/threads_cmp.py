import os, re, subprocess, statistics
H=os.path.expanduser('~'); S=f'{H}/suco_data'; D=f'{H}/deep96_subsets'
CFG=[('deep96_1m',1000000,96,6,16,f'{D}/base_1000000.fvecs',f'{D}/queries.fvecs',f'{D}/gt_1000000.ivecs'),
     ('sift_1m',1000000,128,8,16,f'{H}/sift/sift_base.fvecs',f'{H}/sift/sift_query.fvecs',f'{H}/sift/sift_groundtruth.ivecs'),
     ('gist_250k',250000,960,8,120,f'{H}/gist_sub/base.fvecs',f'{H}/gist_sub/query.fvecs',f'{H}/gist_sub/gt.ivecs')]
def go(c,cwd,env):
    o=subprocess.run(c,cwd=cwd,capture_output=True,text=True,env=env,timeout=3600); o=o.stdout+o.stderr
    m=re.search(r'k = 50, \(recall, ratio\) = \(([0-9.]+)',o); t=re.search(r'query time is ([0-9.]+)',o)
    return (float(m.group(1)),float(t.group(1))) if m and t else (None,None)
print(f"{'dataset':<11}{'mode':<22}{'SuCo recall / ms':<20}{'TaCo recall / ms'}")
for name,n,d,ns,sd,base,q,g in CFG:
    for mode,extra,env in (('1 thread',[],{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','ASC_THREADS':'1'}),
                           ('8 threads',['--parallel-query'],{k:v for k,v in os.environ.items() if k not in ('OMP_NUM_THREADS','ASC_THREADS','OPENBLAS_NUM_THREADS')})):
        sc=['./suco_base','--dataset-path',f'{S}/{name}/base.bin','--query-path',f'{S}/{name}/query.bin','--groundtruth-path',f'{S}/{name}/gt.bin','--dataset-size',str(n+100),'--query-size','1000','--k-size','50','--data-dimensionality',str(d),'--subspace-dimensionality',str(sd),'--subspace-num',str(ns),'--candidate-ratio','0.01','--collision-ratio','0.04','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/sv_idx_suco_{name}','--load-index']+extra
        tc=['./taco_base','--dataset-path',base,'--query-path',q,'--groundtruth-path',g,'--dataset-size',str(n+100),'--query-size','1000','--k-size','50','--data-dimensionality',str(d),'--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio','0.01','--collision-ratio','0.04','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/sv_idx_taco_{name}','--load-index']
        s=[go(sc,f'{H}/SuCo',env) for _ in range(3)]; t=[go(tc,f'{H}/TaCo',env) for _ in range(3)]
        f=lambda r:f"{r[-1][0]:.4f} / {statistics.median(x[1] for x in r):.3f}" if r[-1][0] is not None else "FAILED"
        print(f"{name:<11}{mode:<22}{f(s):<20}{f(t)}",flush=True)
