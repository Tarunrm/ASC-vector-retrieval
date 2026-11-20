import numpy as np, subprocess, os, shutil
H=os.path.expanduser('~'); D=f'{H}/deep96_subsets'; n,dim=1000000,96; NQ=1000
X=np.fromfile(f'{D}/base_1000000.fvecs',dtype=np.float32,count=n*(dim+1)).reshape(n,dim+1)[:,1:]
Q=np.fromfile(f'{D}/queries.fvecs',dtype=np.float32,count=NQ*(dim+1)).reshape(NQ,dim+1)[:,1:]
G=np.fromfile(f'{D}/gt_1000000.ivecs',dtype=np.int32).reshape(-1,101)[:NQ,1:]
def run(binary,extra,tag):
    cmd=[f'./{binary}','--dataset-path',f'{D}/base_1000000.fvecs','--query-path',f'{D}/queries.fvecs','--groundtruth-path',f'{D}/gt_1000000.ivecs','--dataset-size',str(n+100),'--query-size',str(NQ),'--k-size','50','--data-dimensionality','96','--subspace-dimensionality','6','--subspace-num','6','--candidate-ratio','0.005','--collision-ratio','0.04','--kmeans-num-centroid','50','--kmeans-num-iters','2','--index-path',f'{H}/abl_idx_Deep96-1M','--load-index']+extra
    subprocess.run(cmd,cwd=f'{H}/TaCo',capture_output=True,text=True)
    p=f'{H}/dump_{tag}.bin'; shutil.copy(f'{H}/taco_results.bin',p)
    a=np.fromfile(p,dtype=np.int32); return a[2:].reshape(a[0],a[1])
base=run('taco_base',[],'base'); imp2=run('taco_base',['--K-c','100','--T-c','10'],'imp2')
rk=np.zeros(3,int); gap=np.zeros(3,int); tot_missed=0; perfect=0; lost=0; lost_q=0
for i in range(NQ):
    true=G[i][:50]; ts=set(true.tolist()); bf=set(base[i].tolist()); pf=set(imp2[i].tolist())
    missed=[m for m in true if m not in bf]
    if not missed: perfect+=1
    fp=bf-ts
    wr=max(np.sum((X[list(fp)]-Q[i])**2,axis=1)) if fp else None
    for m in missed:
        tot_missed+=1; r=int(np.where(true==m)[0][0]); rk[0 if r<10 else 1 if r<30 else 2]+=1
        if wr:
            g=np.sum((X[m]-Q[i])**2)/wr; gap[0 if g<0.9 else 1 if g<0.98 else 2]+=1
    l=len((bf&ts)-pf); lost+=l; lost_q+= l>0
rec=lambda R:np.mean([len(set(R[i].tolist())&set(G[i][:50].tolist()))/50 for i in range(NQ)])
print(f"Recall@50 baseline {rec(base):.4f}, Imp2 {rec(imp2):.4f}")
print(f"Baseline: perfect queries {perfect}/{NQ}; missed neighbours {tot_missed} ({tot_missed/NQ:.2f}/query)")
print(f"  missed by true rank  1-10: {rk[0]}, 11-30: {rk[1]}, 31-50: {rk[2]}")
print(f"  missed dist / worst returned false-positive dist  <0.9: {gap[0]}, 0.9-0.98: {gap[1]}, >=0.98 (near-tie): {gap[2]}")
print(f"Imp2: neighbours found by baseline but lost by pruning: {lost} across {lost_q} queries")
