import gc
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

####################################################################
#####       METHODS RELATED TO FILE READING & PCA
####################################################################

def read_file(file_path):
    df = pd.read_csv(file_path, header=[0], dtype=np.int32)
    alert_id = None
    if file_path.endswith("data.csv"):
        alert_id = df['alert_ids_int'].values
    features = df.drop('alert_ids_int', axis=1, errors='ignore').values
    return features, alert_id

def read_data_type(dataset_path, data_type):
    data, alert_id = read_file(dataset_path + data_type + '_data.csv')
    label, _ = read_file(dataset_path + data_type + '_label.csv')
    return {'data': data, 'alert_id': alert_id, 'labels': label}

def read_dataset(dataset_path, normalize = True):
    data_dict = {}
    data_dict['train'] = read_data_type(dataset_path, 'train')
    data_dict['val'] = read_data_type(dataset_path, 'val')
    data_dict['test'] = read_data_type(dataset_path, 'test')
    if normalize:
        data_dict['train']['data'] = stats.zscore(data_dict['train']['data'])
        data_dict['val']['data'] = stats.zscore(data_dict['val']['data'])
        data_dict['test']['data'] = stats.zscore(data_dict['test']['data'])
    return data_dict

def reduce_dimension(data_dict, nComp = 10):
    pca = PCA(n_components = nComp)
    pca.fit(data_dict['train']['data'])
    l = pca.explained_variance_ratio_
    data_dict['train']['data'] = pca.transform(data_dict['train']['data'])
    data_dict['val']['data'] = pca.transform(data_dict['val']['data'])
    data_dict['test']['data'] = pca.transform(data_dict['test']['data'])
    return data_dict, sum(l)

def expand_dict(data_dict, x):
    return data_dict[x]['data'], data_dict[x]['labels'], data_dict[x]['alert_id']


####################################################################
#####       METHODS RELATED TO EVALUATION
####################################################################

def evaluate_acc_f1(actual, prediction):
    def get_val(s):
        s = [x for x in s.strip(" \n").split(" ") if len(x) > 0]
        return float(s[-2])
    
    report = classification_report(actual, prediction)
    report = [x.strip(" ") for x in report.split("\n") if len(x.strip(" ")) > 0]
    f1 = get_val(report[-1])
    acc = get_val(report[-3])
    return acc, f1

def plot_roc(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC Curve')
    display.plot()  
    plt.show()   

def evaluate_roc_auc(actual, prediction):
    plot_roc(actual, prediction)
    return roc_auc_score(actual, prediction)

def evaluate(actual, prediction, alert_id = None):
    if alert_id is not None:
        ac = {}
        for x, y in zip(actual, alert_id):
            ac[y] = x
        pr = {}
        for x, y in zip(prediction, alert_id):
            pr[y] = x
        actual, prediction = [], []
        for x in ac:
            actual.append(ac[x])
            prediction.append(pr[x])
    acc, f1 = evaluate_acc_f1(actual, prediction)
    roc_auc = evaluate_roc_auc(actual, prediction)
    return [acc, f1, roc_auc]

def filter_predictions(alert_id, predictions):
    d = {}
    for x in alert_id:
        d[x] = 0
    for x, pred in zip(alert_id, predictions):
        if pred == 1:
            d[x] += 1
        else:
            d[x] -= 1
    for i in range(len(predictions)):
        x = alert_id[i]
        if d[x] > 0:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions


####################################################################
#####       METHODS RELATED TO PERMUTATION OF HYPERPARAMETERS
####################################################################

def permute_hyperparams(ls):
    l = list(itertools.product(*ls))
    l = [list(x) for x in l]
    return l


####################################################################
#####       MAIN METHODS THAT RUN EVERYTHING
####################################################################

def run_model(classifier_class, data_dict, params, verbose = False):
    train_x, train_y, _ = expand_dict(data_dict, 'train')
    val_x, val_y, val_alert = expand_dict(data_dict, 'val')
    test_x, test_y, test_alert = expand_dict(data_dict, 'test')
    
    eval_filter = []
    eval_normal = []
    
    for param in params:
        classifier = classifier_class(param)
        classifier.train_model(train_x, train_y)
        val_pred = classifier.get_predictions(val_x)
        temp = evaluate(val_y, val_pred)
        temp.append(param)
        eval_normal.append(temp)
        val_pred = filter_predictions(val_alert, val_pred)
        temp = evaluate(val_y, val_pred, val_alert)
        temp.append(param)
        eval_filter.append(temp)
    
    ls = [eval_normal, eval_filter]
    output = []
    for eval in ls:
        # print eval list
        if verbose:
            for x in eval:
                print(x)
        
        # find best hyperparams from val data
        best_index, best_acc = -1, -1
        for i in range(len(eval)):
            if best_acc < eval[i][0]:
                best_index = i
                best_acc = eval[i][0]
        
        param = eval[best_index][3]
        classifier = classifier_class(param)
        classifier.train_model(train_x, train_y)
        test_pred = classifier.get_predictions(test_x)
        d = {}
        ind = ls.index(eval)
        
        if ind == 0:
            test_eval = evaluate(test_y, test_pred)
        else:
            test_pred = filter_predictions(test_alert, test_pred)
            test_eval = evaluate(test_y, test_pred, test_alert)
        
        d['validation_eval' + str(ind)] = eval[best_index][:3]
        d['params' + str(ind)] = eval[best_index][3]
        d['test_eval' + str(ind)] = test_eval
        output.append(d)
    return output

def run_all_for_one_dataset(classifier_class, params, path, dataset, run_type = "normal", all = False, normalize = True):
    if run_type == "normal":
        data_dict = read_dataset(path + 'data/dataset_' + str(dataset) + '/', normalize)
        print(run_model(classifier_class, data_dict, params, True))
    else:
        if all:
            start = 4
            end = 41
        else:
            start = 20
            end = 21
        all = not all
        for nComp in range(start, end, 6):
            data_dict = read_dataset(path + 'data/dataset_' + str(dataset) + '/', normalize)
            data_dict, pca_cov = reduce_dimension(data_dict, nComp)
            print(nComp, "pca_cov:", pca_cov, run_model(classifier_class, data_dict, params, all))
            del data_dict
            gc.collect()


####################################################################
#####       CLASSES FOR DIFFERENT MODELS
####################################################################

class artificial_neural_network:
    
    def __init__(self, params):
        self.nEpoch = params[0]
        self.model = self.get_model()
    
    def get_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.nEpoch)
    
    def get_predictions(self, test_input):
        probability_model = tf.keras.Sequential([self.model, 
            tf.keras.layers.Softmax()])
        pred = probability_model.predict(test_input)
        pred = [np.argmax(x) for x in pred]
        return pred
    
class naive_bayes:
    
    def __init__(self, params):
        self.var_smoothing = params[0]
        self.model = self.get_model()
    
    def get_model(self):
        model = GaussianNB(var_smoothing = self.var_smoothing)
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y.ravel())
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        return pred
    
class kmeans:
    
    def __init__(self, params):
        self.n = params[0]
        self.model = self.get_model()
        self.train_labels_pred = None
        self.train_labels_org = None
    
    def get_model(self):
        model = MiniBatchKMeans(n_clusters=self.n,  random_state=0, batch_size=1000)
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x)
        self.train_labels_pred = self.model.labels_
        self.train_labels_org = train_y
    
    def get_labels(self, train_clusters, train_y, nCluster, test_clusters):
        cluster_labels = [[0, 0] for i in range(nCluster)]
        for x, y in zip(train_clusters, train_y):
            cluster_labels[int(x)][int(y)] += 1
        cluster_labels = [np.argmax(x) for x in cluster_labels]
        return [cluster_labels[int(x)] for x in test_clusters]

    def get_predictions(self, test_input):
        pred_cluster = self.model.predict(test_input)
        pred = self.get_labels(self.train_labels_pred, self.train_labels_org, self.n, pred_cluster)
        return pred

class logistic_regression:
    
    def __init__(self, params):
        self.penalty = params[0]
        self.model = self.get_model()
    
    def get_model(self):
        model = LogisticRegression(penalty=self.penalty, max_iter=1000, random_state=0)
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y.ravel())
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        return pred

class decision_tree:
    
    def __init__(self, params):
        self.criterion = params[0]
        self.splitter = params[1]
        self.model = self.get_model()
    
    def get_model(self):
        model = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter, random_state=0)
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y.ravel())
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        return pred

class random_forest:
    
    def __init__(self, params):
        self.criterion = params[0]
        self.n_estimators = params[1]
        self.model = self.get_model()
    
    def get_model(self):
        model = RandomForestClassifier(criterion=self.criterion, n_estimators=self.n_estimators, random_state=0)
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y.ravel())
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        return pred

class adaboost:

    def __init__(self, params):
        self.lr = params[0]
        self.n_estimators = params[1]
        self.model = self.get_model()
    
    def get_model(self):
        model = AdaBoostClassifier(learning_rate=self.lr, n_estimators=self.n_estimators, random_state=0)
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y.ravel())
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        return pred

class catboost:

    def __init__(self, params):
        self.n_estimators = params[0]
        self.model = self.get_model()
    
    def get_model(self):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=self.n_estimators)
        return model

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y.ravel())
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        return pred

class lgbm:

    def __init__(self, params):
        self.params = {}
        self.params['learning_rate']= params[0]
        self.params['boosting_type'] = 'gbdt'
        self.params['objective'] = 'binary'
        self.params['metric'] = 'binary_logloss'
        self.params['max_depth'] = 10
        self.model = self.get_model()
    
    def get_model(self):
        return None

    def train_model(self, train_x, train_y):
        import lightgbm as lgb
        d_train = lgb.Dataset(train_x, label = train_y.ravel())
        self.model = lgb.train(self.params, d_train, 100)
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        pred = (pred > 0.5)
        return pred

class xg_boost:

    def __init__(self, params):
        self.max_depth = params[0]
        self.model = self.get_model()
    
    def get_model(self):
        return None

    def train_model(self, train_x, train_y):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(max_depth=self.max_depth, n_jobs=1).fit(train_x, train_y.ravel())
    
    def get_predictions(self, test_input):
        pred = self.model.predict(test_input)
        return pred
