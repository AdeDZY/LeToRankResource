
import numpy as np
import cPickle
import sys
import re
import os
from sklearn.preprocessing import StandardScaler

# constant and input files 
nshard = 123
basedir = "data/cent1-split-new/cwb-test/"
feat_names = cPickle.load(open(basedir + "feat_names.pkl"))
feat = cPickle.load(open(basedir + "test_feat.pkl")) # 2-D list. feat[q][s]: query q and shard s
truth = cPickle.load(open(basedir + "test_truth.pkl"))
qlen = cPickle.load(open(basedir + "test_qlen.pkl"))


# read data from trained model: shards prior, scaler
scaler = cPickle.load(open(basedir + "scaler.pkl", 'rb'))


# start testing
shardlim = 4
p = 0
nfound_total = 0
nfound_total_lm = 0
nfound_total_taily = 0
pred_sort_all = {}
has_rel = [[i for i in range(nshard) if truth[q][i] > 0] for q in range(200)]
test_queries = [i for i in range(200)]
if True:  

    # 1. filter out empty queries
    for q in test_queries:
        if qlen[q] > 0:
            tmp.append(q)
    test_queries = tmp
    
    # 2. turn the feat 2-D list into numpy matrix
    # 3. scale the feature matrix (mean-variance scaler)
    X_test = []
    Y_test = []
    mem_test =[]
    
    for q in test_queries:
        feat_q = [feat[q][s] for s in range(nshard)]
        feat_q = scaler.transform(feat_q)
        for s in range(nshard):
            X_test.append(feat_q[s])
            Y_test.append(truth[q][s] )
            mem_test.append((q, s))
    
    # 4. write the normalized matrix into rankSVM format
    testFeatFile = open("test_cwb.feat",'w')
    for i, f in enumerate(X_test):
        qid = mem_test[i][0]+1   
        res = "0"
        res += " "
        res += "qid:"+str(qid) + " ";
        for j in range(len(f)):
            res += "{0}:{1} ".format(j + 1, f[j])
        res += "\n"
        testFeatFile.write(res)
    testFeatFile.close()
    
    # 5. call rankSVM. prediction results written in "predictions_cwb"
    import os
    stream = os.popen("/Users/zhuyund/Documents/11642-SearchEngines/svm_rank/svm_rank_classify test_cwb.feat model_cwb.dat predictions_cwb")

    for line in stream:
        print line
    
    ### The following lines count how many relevant documents are retrieved
    ### by LeToR, rank-s, taily, language model, redde, and bigram model.
    predFile = open("predictions_cwb")
    pred = []
    for line in predFile:
        score = float(line)
        pred.append(score)

    # l2r predictions
    pred_sort = []
    i  = 0
    for q in test_queries:
        scores = pred[i:i + nshard]
        res = sorted([(scores[s], s) for s in range(nshard)], reverse=True)
        pred_sort.append(res)
        pred_sort_all[q] = res
        i += nshard
    
    n_found = 0
    for i, q in enumerate(test_queries):
        for score, shardid in pred_sort[i][0:shardlim]:
            n_found += truth[q][shardid]
    print "l2r: ", n_found
    nfound_total += n_found
   
    # rank-s predictions
    n_found_ranks = 0
    for q in test_queries:
        t = 0
        scores = [(feat[q][s][0], s) for s in range(nshard)]
        res = sorted(scores, reverse=True)
        for score, shardid in res[0:shardlim]:      
            n_found_ranks += truth[q][shardid]
            t += truth[q][shardid]
    print "ranks: ", n_found_ranks
    
    # language model (lm) predictions
    n_found_lm = 0
    for q in test_queries:
        t = 0
        scores = [(feat[q][s][2], s) for s in range(nshard)]
        res = sorted(scores, reverse=True)
        for score, shardid in res[0:shardlim]:
            n_found_lm += truth[q][shardid]
            t += truth[q][shardid]
    print "lm: ", n_found_lm
    nfound_total_lm += n_found_lm
    
    # taily
    n_found_taily = 0
    for q in test_queries:
        t = 0
        scores = [(feat[q][s][-2], s) for s in range(nshard)]
        res = sorted(scores, reverse=True)
        for score, shardid in res[0:shardlim]:
            #print shardid,
            n_found_taily += truth[q][shardid]
            t += truth[q][shardid]
    print "taily: ", n_found_taily
    nfound_total_taily += n_found_taily
    
    # redde predictions
    n_found_redde = 0
    for q in test_queries:
        t = 0
        scores = [(feat[q][s][11], s) for s in range(nshard)]
        res = sorted(scores, reverse=True)
        for score, shardid in res[0:shardlim]:         
            n_found_redde += truth[q][shardid]
            t += truth[q][shardid]
    print "redde: ", n_found_redde
    
    # bigram predictions
    n_found_bi = 0
    for q in test_queries:
        t = 0
        scores = [(feat[q][s][15], s) for s in range(nshard)]
        res = sorted(scores, reverse=True)
        for score, shardid in res[0:shardlim]:         
            n_found_bi += truth[q][shardid]
            t += truth[q][shardid]
    print "bigram: ", n_found_bi

### generate LeToR shard list
for i in range(1, 13): # select 1 to 12 shards.
    with open(basedir + "/aol_l2r_all_{0}.shardlist".format(i), 'w') as fo:
        for q in range(0, 200):
        
            fo.write(str(q + 1) + ' ')

            for score, shardid in pred_sort_all[q][0:i]:
                if score:
                    fo.write(str(shardid + 1) + ' ')
            fo.write('\n')
            
            
