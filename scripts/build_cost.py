import os, subprocess, sys
H=os.path.expanduser('~'); ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
CHILD='''
import numpy as np, time, resource, sys, os
fam=sys.argv[1]; H=os.path.expanduser('~'); D=f'{H}/deep96_subsets'; n,d=1000000,96
X=np.ascontiguousarray(np.fromfile(f'{D}/base_1000000.fvecs',dtype=np.float32).reshape(n,d+1)[:,1:])
t=time.time()
if fam=='hnsw':
    import hnswlib; h=hnswlib.Index(space='l2',dim=d); h.init_index(max_elements=n,ef_construction=200,M=16); h.set_num_threads(1); h.add_items(X)
elif fam=='ivfpq':
    import faiss; faiss.omp_set_num_threads(1); q=faiss.IndexFlatL2(d); p=faiss.IndexIVFPQ(q,d,1024,d//2,8); p.train(X[:300000]); p.add(X)
elif fam.startswith('lsh'):
    import faiss; faiss.omp_set_num_threads(1); l=faiss.IndexLSH(d,int(fam[3:])); r=faiss.IndexRefineFlat(l); r.train(X); r.add(X)
elif fam.startswith('annoy'):
    from annoy import AnnoyIndex; a=AnnoyIndex(d,'euclidean')
    for i in range(n): a.add_item(i,X[i])
    a.build(int(fam[5:]),n_jobs=1)
print(f"{fam:<9} build {time.time()-t:7.1f} s   process peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
'''
print("Deep96-1M, one thread. 'none' is the baseline: the dataset loaded and no index.",flush=True)
for fam in ('none','hnsw','ivfpq','lsh256','lsh1024','annoy10','annoy50'):
    r=subprocess.run([sys.executable,'-c',CHILD,fam],capture_output=True,text=True,env=ENV,timeout=7200); print((r.stdout or r.stderr[-300:]).strip(),flush=True)
