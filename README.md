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

# Evaluation (MAP, NDCG, ect)
1. Shard list will be written into ./data/cwb-test/. 
2. Use fedsearch/run_job_writer.rb and fedsearch/make_runs.sh
3. shardmap: /bos/usr0/zhuyund/fedsearch/data/cent1-qw160-split-new-ext/
4. central: /bos/usr0/zhuyund/fedsearch/data/cwB-split-sdm-nospam-central/
5. qrels: /bos/usr0/zhuyund/fedsearch/data/cwb-[2009-2012].qrels /bos/usr0/zhuyund/fedsearch/data/cwb.qrels 

# TODO:
- upload AOL and MQT training data.



