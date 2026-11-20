import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=os.path.expanduser('~/thesis_figs'); os.makedirs(O,exist_ok=True); plt.rcParams.update({'font.size':9,'figure.dpi':130})
D96={'Exact':[(1.0,8.506)],
 'HNSW':[(.8713,.071),(.9455,.120),(.9798,.230),(.9936,.392)],
 'IVF-PQ':[(.7029,.094),(.8726,.300),(.9151,.998)],
 'LSH + re-ranking':[(.9522,1.486),(.9971,3.316),(.9999,11.311),(.9996,6.800),(1.0,10.248)],
 'Annoy':[(.5333,.087),(.8107,.347),(.9558,1.574),(.9915,6.336),(.6047,.119),(.8646,.383),(.9733,1.616),(.9954,6.023)],
 'SuCo':[(.9492,.993),(.9745,1.506),(.9775,1.493)]}
SIFT={'Exact':[(.9997,12.199)],
 'HNSW':[(.8656,.073),(.9493,.131),(.9859,.240),(.9969,.447)],
 'IVF-PQ':[(.6017,.120),(.8312,.384),(.8965,1.384)],
 'LSH + re-ranking':[(.8027,1.478),(.9591,3.259),(.9969,11.383),(.9926,6.696),(.9995,9.923),(.9997,20.331)],
 'Annoy':[(.4679,.089),(.7772,.353),(.9549,1.566),(.9955,5.985),(.5352,.123),(.8185,.407),(.9666,1.567),(.9963,5.616)],
 'SuCo':[(.9338,1.334),(.9538,1.720),(.9617,1.267),(.9750,1.578),(.9707,1.361),(.9889,1.891),(.9963,2.692),(.9993,4.972)]}
def front(pts):
    p=sorted(pts,key=lambda x:x[1]); out=[]; best=-1
    for r,t in p:
        if r>best: out.append((r,t)); best=r
    return out
mk={'Exact':'k*','HNSW':'o-','IVF-PQ':'^-','LSH + re-ranking':'D-','Annoy':'v-','SuCo':'s-'}
f,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,(title,data) in zip(axs,(('Deep96-1M',D96),('SIFT-1M',SIFT))):
    for k,v in data.items():
        p=front(v) if k in ('Annoy','LSH + re-ranking','SuCo') else sorted(v)
        ax.plot([x[1] for x in p],[x[0] for x in p],mk[k],label=k,markersize=6)
    ax.set_xscale('log'); ax.set_xlabel('ms per query (one thread, log scale)'); ax.set_title(title); ax.set_ylim(.5,1.005); ax.grid(alpha=.3)
axs[0].set_ylabel('recall@50'); axs[0].legend(fontsize=8,loc='lower right')
f.tight_layout(); f.savefig(f'{O}/fig_5_9_families.png',dpi=200); print('saved')
