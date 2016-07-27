
import numpy as np
import cPickle
import sys
import re
import os
from sklearn.preprocessing import StandardScaler

# train LeToR model (with AOL queries)
# dependent on rankSVM 
# input: features, groundtruth, query length
# output:
#     model_cwb.dat: trained model
#     scaler.pkl: training data's mean and variance. to scale testing data


# constant and input files 
nshard = 123
basedir = "./data/cent1-split-new/aol-train/" # TODO: cw and mqt training queries
feat_names = cPickle.load(open(basedir + "feat_names.pkl"))
print "reading files..."
feat = cPickle.load(open(basedir + "train_feat.pkl")) 
truth = cPickle.load(open(basedir + "train_truth.pkl"))
qlen = cPickle.load(open(basedir + "train_qlen.pkl"))



# TODO: append your own features to feat here!
# feat: 2-D list. feat[q][s]: features for [query q, shard s]


# start training
# 1. filter out empty queries, and queries with less than 500 retrieved documents.
# 2. turn the feat 2-D list into numpy matrix
# 3. scale the feature matrix (mean-variance scaler)
# 4. write the normalized matrix into rankSVM format
# 5. call rankSVM. model stored in "model_cwb.dat"

train_queries = range(0, 1000)  

# 1. filter out empty queries, and queries with less than 500 retrieved documents.
tmp = []
for q in train_queries:
    if qlen[q] > 0 and sum(truth[q]) >= 500:
        tmp.append(q)
train_queries = tmp

# 2. turn the feat 2-D list into numpy matrix
X_train = []
X_train_pre = []
Y_train = []
Y_train_pre = []

mem_train = []
    
for q in train_queries:
    feat_q = [feat[q][s] for s in range(nshard)]
    redde_q_sorted = sorted([(val[2],i) for i, val in enumerate(feat[q])], reverse=True)
    i = 0
    for val, s in redde_q_sorted:
        if i < nshard or truth[q][s] > 0:
            X_train.append(feat_q[s])
            Y_train.append(truth[q][s])
            mem_train.append((q, s))
        i += 1

# 3. scale the feature matrix (mean-variance scaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
    
# 4. write the normalized matrix into rankSVM format
print "writing into rankSVM input file train_cwb.feat..."
trainFeatFile = open("train_cwb.feat",'w')
for i, f in enumerate(X_train):
    qid = mem_train[i][0] +1     
    res = ""
    if Y_train[i]:
        res += str(Y_train[i])
    else:
        res += "0"
    res += " qid:"+str(qid) + " ";
    for j in range(len(f)):
        res += "{0}:{1} ".format(j + 1, f[j])
    res += "\n"
    trainFeatFile.write(res)
trainFeatFile.close()

# 5. call rankSVM. model stored in "model_cwb.dat"
import os
print "training..."
stream = os.popen("/Users/zhuyund/Documents/11642-SearchEngines/svm_rank/svm_rank_learn -c 1 -g 0.001 -t 0 train_cwb.feat model_cwb.dat")
print "finished training! model_cwb.dat, scaler.pkl"


# store scaler. Will be used in testing phase.
cPickle.dump(scaler, open("scaler.pkl", 'wb'))


# print model weights
f = open("model_cwb.dat")
lines = f.readlines()
wline = lines[-1]
items = wline.split(' ')
wf = []
for item in items[1:-1]:
    fid, w = item.split(':')
    fid = int(fid)
    w = float(w)
    wf.append((w, feat_names[fid-1]))
for w, name in sorted(wf, reverse=True):
    print w, name

