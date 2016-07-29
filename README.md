# LeToRankResource
Learning to rank resources for selective search

# dependence
- SVM Rank
- Python sklean library

# How to Run:
1. Download /bos/tmp11/zhuyund/LeToRankResource/data.zip. Put under the same dir with source code and unzip.
2. Modify SVMRank path in the source code. Modify training basedir (./data/aol-train/ or ./data/mqt-train/) in pairwise-train-AOL-cleaned.py.
3. python ./pairwise-train-AOL-cleaned.py 
4. python ./pairwise-test-clean.py
5. pairwise-test-clean.py will print out number of relevant documents retrieved by each method (when selecting 4 shards).
6. Shard list will be written into ./data/cwb-test/aol_l2r_all_{1-10}.shardlist. 

# Evaluation (MAP, NDCG, ect)
## Setup
0. Be careful the following files don't override your own ones.
1. `cp /bos/usr0/zhuyund/fedsearch/run_l2r_cent1.sh ~/fedsearch/.` 
2. `cp /bos/usr0/zhuyund/fedsearch/l2r_make_runs.sh ~/fedsearch/.`
3. Copy qrels: `cp /bos/usr0/zhuyund/fedsearch/data/cwb*.qrels  ~/fedsearch/data/.` 

## To test a shard list
4. `mkdir ~/fedsearch/output/rankings/l2r/cent1-qw160-split-new/{runname}`. For example, runname='aoltrain_lim6' means LeToR trained with AOL queries, and search the top 6 shards.
5. Copy the shard list you want to test into ~/fedsearch/output/rankings/l2r/cent1-qw160-split-new/{runname}. `cp ./data/cwb-test/aol_l2r_all_6.shardlist ~/fedsearch/output/rankings/l2r/cent1-qw160-split-new/aoltrain_lim6/all.shardlist`
6. `~/fedsearch/run_l2r_cent1.sh {runname}`
7. TrecEval results will be written into  `~/fedsearch/output/rankings/l2r/cent1-qw160-split-new/{runname}/cwb*.eval(and cwb*.Qeval)`

# TODO:
- upload AOL and MQT training data.



