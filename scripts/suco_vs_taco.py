import subprocess, os, re, statistics
H=os.path.expanduser('~'); S=f'{H}/suco_data'; D=f'{H}/deep96_subsets'
# name, n, dim, suco (subspaces, subspace dim), taco files
CFG=[('deep96_1m',1000000,96,6,16,f'{D}/base_1000000.fvecs',f'{D}/queries.fvecs',f'{D}/gt_1000000.ivecs'),
     ('deep96_2m',2000000,96,6,16,f'{D}/base_2000000.fvecs',f'{D}/queries.fvecs',f'{D}/gt_2000000.ivecs'),
     ('sift_1m',1000000,128,8,16,f'{H}/sift/sift_base.fvecs',f'{H}/sift/sift_query.fvecs',f'{H}/sift/sift_groundtruth.ivecs'),
     ('gist_250k',250000,960,8,120,f'{H}/gist_sub/base.fvecs',f'{H}/gist_sub/query.fvecs',f'{H}/gist_sub/gt.ivecs')]
SETTINGS=[(0.04,0.005),(0.04,0.01),(0.08,0.01)]
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','ASC_THREADS':'1'}
def wall(s):
    p=[float(x) for x in s.split(':')]; return sum(v*60**i for i,v in enumerate(reversed(p)))
def run(cmd,cwd):
    r=subprocess.run(['/usr/bin/time','-v']+cmd,cwd=cwd,capture_output=True,text=True,env=ENV,timeout=7200)
    o=r.stdout+r.stderr
    g=lambda p:(re.search(p,o) or [None,None])[1]
    rec=g(r'k = 50, \(recall, ratio\) = \(([0-9.]+)'); qt=g(r'average query time is ([0-9.]+)')
    it=g(r'indexing time is: ([0-9.]+)'); rss=g(r'Maximum resident set size \(kbytes\): (\d+)'); w=g(r'Elapsed.*: ([0-9:.]+)')
    return dict(rec=float(rec) if rec else None,qt=float(qt) if qt else None,idx=float(it)/1000 if it else None,
                rss=int(rss)/1e6 if rss else None,wall=wall(w) if w else None,tail=o[-300:])
def suco(name,n,d,ns,sd,a,b,idx,nq,load):
    c=[f'./suco','--dataset-path',f'{S}/{name}/base.bin','--query-path',f'{S}/{name}/query.bin','--groundtruth-path',f'{S}/{name}/gt.bin',
       '--dataset-size',str(n+100),'--query-size',str(nq),'--k-size','50','--data-dimensionality',str(d),'--subspace-dimensionality',str(sd),
       '--subspace-num',str(ns),'--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',idx]
    return run(c+(['--load-index'] if load else []),f'{H}/SuCo')
def taco(base,q,g,n,d,a,b,idx,nq,load):
    c=['./taco_base','--dataset-path',base,'--query-path',q,'--groundtruth-path',g,'--dataset-size',str(n+100),'--query-size',str(nq),'--k-size','50',
       '--data-dimensionality',str(d),'--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio',str(b),'--collision-ratio',str(a),
       '--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',idx]
    return run(c+(['--load-index'] if load else []),f'{H}/TaCo')
for name,n,d,ns,sd,base,q,g in CFG:
    print(f"\n=== {name} ({n} x {d}), one thread ===",flush=True)
    si=f'{H}/sv_idx_suco_{name}'; ti=f'{H}/sv_idx_taco_{name}'
    sb=suco(name,n,d,ns,sd,0.04,0.005,si,100,False); tb=taco(base,q,g,n,d,0.04,0.005,ti,100,False)
    print(f"{'build':<7}{'system':<7}{'index stage s':<15}{'wall s':<9}{'peak GB'}")
    for lab,x in (('SuCo',sb),('TaCo',tb)):
        print(f"{'':<7}{lab:<7}{(x['idx'] if x['idx'] is not None else float('nan')):<15.1f}{(x['wall'] or float('nan')):<9.1f}{(x['rss'] or float('nan')):.2f}" if x['wall'] else f"  {lab} FAILED: {x['tail']}",flush=True)
    print(f"{'query':<7}{'system':<7}{'alpha':<7}{'beta':<7}{'recall@50':<11}{'ms/query (median of 3)'}")
    for a,b in SETTINGS:
        for lab,fn in (('SuCo',lambda:suco(name,n,d,ns,sd,a,b,si,1000,True)),('TaCo',lambda:taco(base,q,g,n,d,a,b,ti,1000,True))):
            rs=[fn() for _ in range(3)]; ok=[r for r in rs if r['rec'] is not None]
            print(f"{'':<7}{lab:<7}{a:<7}{b:<7}{ok[-1]['rec']:<11.4f}{statistics.median(r['qt'] for r in ok):.3f}" if ok else f"  {lab} {a} {b} FAILED: {rs[0]['tail']}",flush=True)
