import os, re, subprocess
H=os.path.expanduser('~'); ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
def run(binary,S,n,a,b,idx,nq,load):
    c=['/usr/bin/time','-v',f'./{binary}','--dataset-path',f'{S}/base.bin','--query-path',f'{S}/query.bin','--groundtruth-path',f'{S}/gt.bin','--dataset-size',str(n+100),'--query-size',str(nq),'--k-size','50','--data-dimensionality','960','--subspace-dimensionality','120','--subspace-num','8','--candidate-ratio',str(b),'--collision-ratio',str(a),'--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',idx]+(['--load-index'] if load else [])
    r=subprocess.run(c,cwd=f'{H}/SuCo',capture_output=True,text=True,env=ENV,timeout=10800); o=r.stdout+r.stderr
    g=lambda p:(re.search(p,o) or [None,None])[1]
    return dict(rec=g(r'k = 50, \(recall, ratio\) = \(([0-9.]+)'),qt=g(r'average query time is ([0-9.]+)'),rss=g(r'Maximum resident set size \(kbytes\): (\d+)'),w=g(r'Elapsed.*: ([0-9:.]+)'),tail=o[-200:])
print("--- check on Gist-250K: recall must match the earlier 0.7811 at alpha 0.04 beta 0.01 ---",flush=True)
S=f'{H}/suco_data/gist_250k'; b=run('suco_ref',S,250000,0.04,0.01,f'{H}/ref_idx_250k',100,False)
print("250K build: wall",b['w'],"peak GB",round(int(b['rss'])/1e6,2) if b['rss'] else 'FAILED '+b['tail'],"(was 4.73 GB)",flush=True)
q=run('suco_ref',S,250000,0.04,0.01,f'{H}/ref_idx_250k',1000,True); print("250K recall",q['rec'],"ms",q['qt'],flush=True)
if q['rec'] and abs(float(q['rec'])-0.7811)<0.002:
    print("\n--- Gist 1M ---",flush=True); S=f'{H}/gist_scale/1000000'; n=1000000; idx=f'{H}/ref_idx_1m'
    b=run('suco_ref',S,n,0.04,0.01,idx,100,False)
    if not b['rec']: print("1M build FAILED:",b['tail'])
    else:
        print("1M build: wall",b['w'],"peak GB",round(int(b['rss'])/1e6,2),flush=True)
        for a,bb in ((0.04,0.01),(0.04,0.05),(0.08,0.05),(0.12,0.05)):
            r=run('suco_ref',S,n,a,bb,idx,1000,True); print(f"SuCo alpha={a} beta={bb}: recall {r['rec']}  {r['qt']} ms",flush=True)
else: print("recall changed or missing: stop and check the patch before the 1M run")
