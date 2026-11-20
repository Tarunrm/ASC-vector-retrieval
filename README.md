# ASC-vector-retrieval

Code and raw results for the Master's thesis **"Adaptive Subspace Collision for Efficient Vector Embedding Retrieval in RAG Systems"**
(University of Melbourne, supervisor Dr. Jianzhong Qi).

- `scripts/`: experiment and figure scripts (Appendix A of the thesis)
- `results/`: raw console outputs (every table in Chapter 5 comes from one of these files)
- `source/`: modified TaCo source (`-DASC_IMP1` builds the query-adaptive budget and cluster pruning)
- `figures/`: figures used in the thesis

Datasets (Deep, SIFT, GloVe, NYTimes, Gist, MS MARCO, FiQA) are public and are not stored here.
Experiments ran on WSL2 (Ubuntu), single query thread, about 15 GiB of memory.
