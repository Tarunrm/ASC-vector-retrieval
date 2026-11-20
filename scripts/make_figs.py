import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=os.path.expanduser('~/thesis_figs'); plt.rcParams.update({'font.size':10,'figure.dpi':130})
def save(f,n): f.tight_layout(); f.savefig(f'{O}/{n}.png',dpi=200); plt.close(f)
def front(pts):  # lower-time / higher-recall frontier
    pts=sorted(pts,key=lambda p:p[1]); out=[]; best=-1
    for r,t in pts:
        if r>best: out.append((r,t)); best=r
    return out
# ---- Fig 5.1/5.2 recall-time curves, one thread
suco_d2=[(.9209,1.754),(.9627,3.572),(.9888,7.543),(.9894,8.371),(.9629,2.248),(.9820,3.376),(.9848,4.178),(.9985,15.016),(.9693,2.395),(.9834,3.291),(.9964,5.222),(.9985,9.074),(.9654,2.648),(.9895,3.928),(.9957,5.534),(.9997,10.653)]
hnsw_d2=[(.8514,.086),(.9316,.137),(.9722,.272),(.9909,.452),(.9969,.811)]
taco_d2=[(.9521,5.89),(.9722,5.22),(.9880,6.175)]
suco_s=[(.9337,1.334),(.9538,1.720),(.9630,2.128),(.9950,6.012),(.9617,1.267),(.9750,1.578),(.9950,2.927),(.9966,3.962),(.9707,1.361),(.9889,1.891),(.9963,2.692),(.9990,4.527),(.9701,1.639),(.9888,2.104),(.9970,3.031),(.9993,4.972)]
hnsw_s=[(.8672,.089),(.9498,.143),(.9862,.265),(.9968,.476),(.9994,.866)]
def curve(ax,series,title):
    for lab,pts,st,front_only in series:
        p=front(pts) if front_only else pts
        ax.plot([x[1] for x in p],[x[0] for x in p],st,label=lab)
    ax.set_xscale('log'); ax.set_xlabel('ms per query (one thread, log scale)'); ax.set_ylabel('recall@50'); ax.set_title(title); ax.grid(alpha=.3); ax.legend()
f,a=plt.subplots(figsize=(5.5,4)); curve(a,[('HNSW',hnsw_d2,'o-',False),('SuCo (best of grid)',suco_d2,'s-',True),('TaCo (6x6)',taco_d2,'^',False)],'Deep96-2M'); a.set_ylim(.85,1.0); save(f,'fig_5_1_curves_deep96_2m')
f,a=plt.subplots(figsize=(5.5,4)); curve(a,[('HNSW',hnsw_s,'o-',False),('SuCo (best of grid)',suco_s,'s-',True)],'SIFT-1M'); a.set_ylim(.85,1.0); save(f,'fig_5_2_curves_sift1m')
# ---- Fig 5.3 Gist: HNSW vs corpus size, SuCo 500K, TaCo 250K
f,a=plt.subplots(figsize=(6,4.2))
H={'HNSW 250K':[(.8436,.386),(.9279,.670),(.9697,1.171),(.9877,2.040)],'HNSW 500K':[(.8036,.506),(.8969,.894),(.9524,1.502),(.9786,2.662)],'HNSW 1M':[(.7555,.741),(.8589,1.098),(.9281,1.941),(.9650,3.391)]}
for (k,v),s in zip(H.items(),['o-','s-','^-']): a.plot([x[1] for x in v],[x[0] for x in v],s,label=k)
a.plot([x[1] for x in [(.81634,3.533),(.9466,9.773),(.9725,10.764),(.9804,11.178)]],[.81634,.9466,.9725,.9804],'k*--',label='SuCo 500K')
a.plot([2.01,3.18,6.19,8.02],[.8429,.8996,.9745,.9811],'kD:',label='TaCo 250K (alpha 0.2, beta sweep)')
a.set_xscale('log'); a.set_xlabel('ms per query (one thread, log scale)'); a.set_ylabel('recall@50'); a.set_title('Gist'); a.grid(alpha=.3); a.legend(fontsize=8); save(f,'fig_5_3_gist')
# ---- Fig 5.4 ablation deltas (TaCo)
ds=['Deep96\n100K','Deep96\n500K','Deep96\n1M','Deep96\n2M','SIFT\n1M','GloVe','NYTimes']
d1=[.3,-.2,-.3,-.1,-2.5,-.05,-3.4]; d2=[-3.1,-2.4,-2.4,-1.9,-1.3,-5.6,-3.9]; d12=[-2.8,-2.6,-2.8,-2.1,-3.7,-5.6,-5.6]
x=np.arange(7); w=.27; f,a=plt.subplots(figsize=(7.5,4))
a.bar(x-w,d1,w,label='Query-adaptive budget'); a.bar(x,d2,w,label='Cluster pruning'); a.bar(x+w,d12,w,label='Budget + pruning')
a.axhline(0,color='k',lw=.8); a.set_xticks(x); a.set_xticklabels(ds,fontsize=8); a.set_ylabel('change in recall@50 (points)'); a.set_title('Ablation on TaCo, against baseline'); a.legend(); a.grid(axis='y',alpha=.3); save(f,'fig_5_4_ablation')
# ---- Fig 5.5 build cost and memory
nm=['Deep96-1M','Deep96-2M','SIFT-1M','Gist-250K']; sw=[10.5,21.0,13.8,13.9]; tw=[19.0,38.3,28.2,234.2]; sm=[2.00,3.99,2.65,4.73]; tm=[3.39,6.77,4.51,8.45]
f,(a,b)=plt.subplots(1,2,figsize=(9,3.8)); x=np.arange(4)
a.bar(x-.2,sw,.4,label='SuCo'); a.bar(x+.2,tw,.4,label='TaCo'); a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels(nm,fontsize=8); a.set_ylabel('build wall-clock (s, log)'); a.legend(); a.set_title('Build time, one thread')
b.bar(x-.2,sm,.4,label='SuCo'); b.bar(x+.2,tm,.4,label='TaCo'); b.set_xticks(x); b.set_xticklabels(nm,fontsize=8); b.set_ylabel('peak memory (GB)'); b.set_title('Peak memory'); save(f,'fig_5_5_build')
# ---- Fig 5.6 miss rate by rank
f,a=plt.subplots(figsize=(4.8,3.6)); a.bar(['1-10','11-30','31-50'],[3.1,6.4,8.8],color='tab:gray'); a.set_xlabel('true rank of the neighbour'); a.set_ylabel('miss rate (%)'); a.set_title('Baseline misses, Deep96-1M'); save(f,'fig_5_6_miss_by_rank')
# ---- Fig 5.7 Query-adaptive budget: TaCo-Gist vs SuCo-Gist, difference with 95% interval
f,(a,b)=plt.subplots(1,2,figsize=(9,3.8),sharey=False)
al=[.02,.04,.06,.08]; dt=np.array([4.65,2.55,6.09,-.63]); lo=np.array([4.20,2.27,5.46,-1.06]); hi=np.array([5.11,2.84,6.70,-.21])
a.errorbar(range(4),dt,yerr=[dt-lo,hi-dt],fmt='o',capsize=4); a.set_xticks(range(4)); a.set_xticklabels([f'a={v}\ntime x{t}' for v,t in zip(al,[.99,1.00,1.48,1.13])],fontsize=8); a.axhline(0,color='k',lw=.8); a.set_title('TaCo, Gist-250K (beta 0.05)'); a.set_ylabel('Query-adaptive budget minus baseline, recall@50 (points)')
al2=[.02,.04,.08]; ds2=np.array([-.35,-.59,-.64]); l2=np.array([-.70,-1.03,-1.08]); h2=np.array([-.02,-.17,-.20])
b.errorbar(range(3),ds2,yerr=[ds2-l2,h2-ds2],fmt='o',color='tab:red',capsize=4); b.set_xticks(range(3)); b.set_xticklabels([f'a={v}\ntime x{t}' for v,t in zip(al2,[1.02,1.02,1.01])],fontsize=8); b.axhline(0,color='k',lw=.8); b.set_title('SuCo, Gist-250K (beta 0.01)'); b.set_ylabel('Query-adaptive budget minus baseline, recall@50 (points)'); save(f,'fig_5_7_imp1_gist')
# ---- Fig 5.8 end-to-end nDCG difference vs exact search
nm=['HNSW','SuCo','SuCo+Query-adaptive budget','TaCo','ASC (TaCo)']; d=np.array([-.46,-.17,-.16,-1.03,-3.31]); l=np.array([-1.15,-.66,-.64,-2.03,-4.83]); h=np.array([.08,.16,.19,-.12,-1.85])
f,a=plt.subplots(figsize=(6,3.8)); a.errorbar(range(5),d,yerr=[d-l,h-d],fmt='o',capsize=4); a.axhline(0,color='k',lw=.8); a.set_xticks(range(5)); a.set_xticklabels(nm,fontsize=8); a.set_ylabel('nDCG@10 after re-ranking\nminus exact search (points)'); a.set_title('First stage and end-to-end ranking, MS MARCO'); a.grid(axis='y',alpha=.3); save(f,'fig_5_8_end_to_end')
print(sorted(os.listdir(O)))
