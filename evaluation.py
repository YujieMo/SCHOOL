from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score
import numpy as np
from munkres import Munkres
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


class evaluation_metrics():
    def __init__(self, embs, labels, args, train_idx=None, val_idx=None, test_idx=None):

        self.embs = embs
        self.args = args
        if args.dataset in ['Aminer','photo','computers', 'cs', 'physics', 'cora', 'citeseer', 'pubmed']:
            self.trX, self.trY = self.embs[train_idx], np.array(labels[train_idx])
            self.valX, self.valY = self.embs[val_idx], np.array(labels[val_idx])
            self.tsX, self.tsY = self.embs[test_idx], np.array(labels[test_idx])
            self.n_label = int(max(labels)-min(labels)+1)
        else:
            train, val, test = labels
            self.trX, self.trY = self.embs[np.array(train)[:,0]], np.array(train)[:,1]
            self.valX, self.valY = self.embs[np.array(val)[:,0]], np.array(val)[:,1]
            self.tsX, self.tsY = self.embs[np.array(test)[:,0]], np.array(test)[:,1]
            self.n_label = len(set(self.tsY))
    def evalutation(self, args):
        fis, fas = 0,0
        for rs in [0, 1, 2, 3, 4]:
            lr = LogisticRegression(max_iter=500, random_state=rs, solver='sag')
            lr.fit(self.trX, self.trY)
            Y_pred = lr.predict(self.tsX)
            f1_micro = metrics.f1_score(self.tsY, Y_pred, average='micro')
            f1_macro = metrics.f1_score(self.tsY, Y_pred, average='macro')
            fis+=f1_micro
            fas+=f1_macro
        print('\t[Classification] f1_macro=%.5f, f1_micro=%.5f' % (fas/5, fis/5))

        # kmeans = KMeans(n_clusters=self.n_label, random_state=0).fit(self.valX)
        # preds = kmeans.predict(self.trX)
        # nmi = metrics.normalized_mutual_info_score(labels_true=self.trY, labels_pred=np.array(preds))
        # print('\t[Clustering] nmi=%.5f' % (nmi))

        k1 = run_kmeans(self.tsX, self.tsY, self.n_label)
        k1[0] = float(k1[0])
        k1[1] = float(k1[1])
        print('\t[Clustering] acc, nmi, ari: {:.4f} | {:.4f} | {:.4f} '.format(k1[0], k1[1], k1[2]))
        st = run_similarity_search(self.tsX, self.tsY)
        print("\t[Similarity] [5,10,20,50,100] : {}".format(st))

        return fis / 5, fas / 5, k1, st

    # def evaluate_cluster(self, args):
    #     kmeans = KMeans(n_clusters=self.n_label, random_state=0).fit(self.valX)
    #     preds = kmeans.predict(self.trX)
    #     nmi = metrics.normalized_mutual_info_score(labels_true=self.trY, labels_pred=np.array(preds))
    #     print('\t[Clustering] nmi=%.5f' % (nmi))
    #     return nmi

def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k, algorithm="elkan")

    NMI_list2 = []
    for i in range(1):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        # s = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        cm = clustering_metrics(y, y_pred)
        acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
        NMI_list2.append(acc)
        NMI_list2.append(nmi)
        NMI_list2.append(adjscore)

    # mean = np.mean(NMI_list)
    # std = np.std(NMI_list)
    # print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    # print('\t[Clustering] acc, nmi, ari: {:.4f} | {:.4f} | {:.4f} '.format(acc, nmi, adjscore))
    return NMI_list2




def run_similarity_search(test_embs, test_lbls):

    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]: #
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))

    st = ','.join(st)
    # print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))
    # if not st == None:
    #     st = st.split(', ')
    return st


class clustering_metrics():
    "from https://github.com/Ruiqi-Hu/ARGA"

    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, adjscore


