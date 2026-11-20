import numpy as np, os
H=os.path.expanduser('~'); out=f'{H}/suco_data'
D=f'{H}/deep96_subsets'
CFG=[('deep96_1m',f'{D}/base_1000000.fvecs',f'{D}/queries.fvecs',f'{D}/gt_1000000.ivecs',1000000,96),
     ('deep96_2m',f'{D}/base_2000000.fvecs',f'{D}/queries.fvecs',f'{D}/gt_2000000.ivecs',2000000,96),
     ('sift_1m',f'{H}/sift/sift_base.fvecs',f'{H}/sift/sift_query.fvecs',f'{H}/sift/sift_groundtruth.ivecs',1000000,128),
     ('gist_250k',f'{H}/gist_sub/base.fvecs',f'{H}/gist_sub/query.fvecs',f'{H}/gist_sub/gt.ivecs',250000,960)]
def fv(path,n,d):
    a=np.fromfile(path,dtype=np.float32,count=n*(d+1)).reshape(n,d+1)
    assert (a[:,0].view(np.int32)==d).all(), path
    return np.ascontiguousarray(a[:,1:])
for name,b,q,g,n,d in CFG:
    os.makedirs(f'{out}/{name}',exist_ok=True)
    fv(b,n,d).tofile(f'{out}/{name}/base.bin')
    fv(q,1000,d).tofile(f'{out}/{name}/query.bin')
    G=np.fromfile(g,dtype=np.int32).reshape(-1,101)[:1000,1:51].astype(np.int64)
    assert G.max()<n, name
    G.tofile(f'{out}/{name}/gt.bin')
    print(name,'done',n,d,G.shape)
