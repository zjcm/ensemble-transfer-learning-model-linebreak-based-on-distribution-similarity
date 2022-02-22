import logging
import scipy.stats
from scipy.stats import ks_2samp
from scipy.stats import gamma
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics
import numpy as np
from numpy import e
# logging.basicConfig(filename='test.log', level=logging.DEBUG) # no use

# distribution

# compute JS_divergence for two vectors.
def _JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.ks_2samp(p, M).statistic+0.5*scipy.stats.ks_2samp(q, M).statistic

# compute JS_divergence for two matrixs, by vectors respectively.
def muti_js_micro(M1,M2):
    M1 = M1.copy()
    M2 = M2.copy()
    M1 = M1.reset_index(drop=True)
    M2 = M2.reset_index(drop=True)
    distance = 0
    col_num = M1.shape[1]
    if M2.shape[1] != col_num:
        raise ValueError("column number dos not match. ({0} != {1})".format(M1.shape[1],M2.shape[1]))
    for i in range(col_num):
        c1,c2 = M1.iloc[:, i],M2.iloc[:, i]
        score = _JS_divergence(c1,c2)
        #score = ks_2samp(c1, c2).statistic
        distance += score ** 2
    distance = distance / col_num
    return distance

# mmd 准则
def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    X,Y = X.values, Y.values
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# HSIC 准则
def rbf_dot(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
	H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

	Q = np.tile(G, (1, size2[0]))
	R = np.tile(H.T, (size1[0], 1))

	H = Q + R - 2* np.dot(pattern1, pattern2.T)

	H = np.exp(-H/2/(deg**2))

	return H

def hsic_gam(X, Y, alph = 0.5):
	"""
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""

	X,Y = X.values, Y.values

	n = X.shape[0]

	# ----- width of X -----
	Xmed = X

	G = np.sum(Xmed*Xmed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Xmed, Xmed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	# ----- width of X -----
	Ymed = Y

	G = np.sum(Ymed*Ymed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Ymed, Ymed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	bone = np.ones((n, 1), dtype = float)
	H = np.identity(n) - np.ones((n,n), dtype = float) / n

	K = rbf_dot(X, X, width_x)
	L = rbf_dot(Y, Y, width_y)

	Kc = np.dot(np.dot(H, K), H)
	Lc = np.dot(np.dot(H, L), H)

	testStat = np.sum(Kc.T * Lc) / n

	varHSIC = (Kc * Lc / 6)**2

	varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

	varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

	K = K - np.diag(np.diag(K))
	L = L - np.diag(np.diag(L))

	muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
	muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

	mHSIC = (1 + muX * muY - muX - muY) / n

	al = mHSIC**2 / varHSIC
	bet = varHSIC*n / mHSIC

	thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

	return 1 / testStat

# A-distance准则 略作修改，未使用Hinge损失而是使用了accuracy
from sklearn.linear_model import LogisticRegression
def A_distance(a, b):
    a,b = a.values,b.values
    N = len(a)
    X = np.concatenate([a, b])
    Y = [0] * N + [1] * N

    m = LogisticRegression()
    m.fit(X, Y)
    Yhat = m.predict(X)
    ans = 0
    for i, c in enumerate(Yhat):
        if c == Y[i]:
            ans += 1
    return ans / (2 * N)


# prediction_pipeline
def prediction_pipeline(d,model=None,fitted=None,output="model",task_rows=None,
                        base_learner=None,distance_metrics = None, verbose = False,
                        promodel_score=None,promodel_c = 0,ask_scores=False):
    Y = d["Y"]
    X = d.drop(["Y"], axis=1)
    # training stage
    if output == "model":
        if model is None: # you must assgin a model type
            raise ValueError("param model must be a prediction class.")
        model = model()
        if type(model) == ProModel:
            model._train_rows = task_rows
            model.verbose = verbose
            model.base_learner = base_learner
            model.distance_metrics = distance_metrics

            if promodel_score:
                model.promodel_score = promodel_score
            if promodel_c is not None:
                model.promodel_c = promodel_c

        model.fit(X, Y)
        if ask_scores:
            return model._scores,model
        else:
            return model

    # tesing scene
    if output == "y_hat":
        if fitted is None: # you must assgin a fitted model
            raise ValueError("param fitted must be a model.")
        y_hat = fitted.predict(X)
        return y_hat,mse(Y,y_hat)


# proposed model
# base learner
class ProModel():
    def __init__(self):
        self.task_num = 1
        self.base_learner = None
        self.distance_metrics = muti_js_micro
        self.verbose = 0
        # ==
        self._base_learners = []
        self._X = []
        self._scores = []
        self._train_rows = None
        self._test_rows = None
        # ==
        self.promodel_score = None
        self.promodel_c = None

    def fit(self,X,Y,base_learner=None):
        np.random.seed(0)
        batchs_X,batchs_Y = ProModel.make_batch(X,Y,self._train_rows)
        batch_num = len(Y) // self._train_rows
        self.task_num = batch_num
        if base_learner is None:
            base_learner = LinearRegression
        for i in range(batch_num):
            X = batchs_X[i]
            Y = batchs_Y[i]
            l = base_learner()
            l.fit(X,Y)
            self._base_learners.append(l)
        self._X = batchs_X.copy()
        return self

    def predict(self,X):
        np.random.seed(0)
        _outputs = [] 
        for i in range(self.task_num):
            self._scores.append(self.distance_metrics(X,self._X[i]))
            _outputs.append(self._base_learners[i].predict(X))
        if 1 and self.verbose:
            print(" -- before -- ")
            print(self._scores)
        self._scores = ProModel.score_conversion(self._scores,c=self.promodel_c)
        if 1 and self.verbose:
            print(" -- after -- ")
            print(self._scores)
        if self.promodel_score:
            if self.promodel_score == "even":
                self._scores = [1/self.task_num for i in range(self.task_num)]
                if 1 and self.verbose:
                    print("debug: even test done")
                    print(self._scores)
        output = np.zeros(len(X))
        for i,c in enumerate(_outputs):
            output = output + self._scores[i] * _outputs[i]

        if 0 and self.verbose:
            print(">" * 30)
            for i in _outputs:
                print(i[-5:])
            print(output[-5:])
            print(">" * 30)
        return output

    @staticmethod
    def score_conversion(scores,c=0):
        if c <= -1: c = -0.8
        if c >= 1: c = 0.8
        size = len(scores)
        mv = min(scores)
        scores = [x - c * mv for x in scores]
        scores = [1 / x for x in scores]
        scores = [e ** (x / (size ** 0.5)) for x in scores]
        #scores = [e ** (1 / (size ** 0.5) * x) for x in scores]
        s = sum(scores)
        scores = [x / s for x in scores]
        return scores

    @staticmethod
    def make_batch(X,Y=None, batch_rows=None):
        resX = []
        resY = []
        #values = d.values
        total_rows = len(X)
        if batch_rows is None:
            batch_rows = total_rows
        batch_num = total_rows // batch_rows
        for i in range(batch_num):
            offset = i * batch_rows
            start, end = offset, offset + batch_rows
            # make X
            buffer = X[start:end].copy()
            buffer = buffer.reset_index(drop=True)
            if "Y" in buffer.columns.tolist():
                buffer = buffer.drop(["Y"], axis=1)
            resX.append(buffer)
            # make Y
            if Y is not None:
                buffer = Y[start:end].copy()
                buffer = buffer.reset_index(drop=True)
                resY.append(buffer)
        return resX, resY


