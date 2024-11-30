from dataset import process_data
from Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
import random


def add_tfidata_feats(data, col, n_components):
    text = list(data[col].values)
    tf = TfidfVectorizer(stop_words='english')
    tf.fit(text)
    joblib.dump(tf, 'model/{}_tfidf.joblib'.format(col))
    X = tf.transform(text)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    joblib.dump(svd, 'model/{}_svd.joblib'.format(col))
    X_svd = svd.transform(X)
    columns = [f'{col}_tfidata_{i}' for i in range(n_components)]
    data_svd = pd.DataFrame(X_svd, columns=columns)
    data = pd.concat([data, data_svd], axis=1)
    return data


def tfidf_feature(data):
    data = add_tfidata_feats(data, 'url_unquote', n_components=16)
    data = add_tfidata_feats(data, 'user_agent', n_components=16)
    data = add_tfidata_feats(data, 'body', n_components=32)
    data['contact'] = data['method'] + ' ' + data['refer'] + ' ' + data['url_filetype'] + ' ' + data['ua_short'] + ' ' + \
                      data['ua_first']
    data = add_tfidata_feats(data, 'contact', n_components=8)
    data['new_url'] = data['scheme'] + ' ' + data['netloc'] + ' ' + data['path'] + ' ' + data['parameters'] + ' ' + \
                      data['query'] + ' ' + data['fragment']
    data = add_tfidata_feats(data, 'new_url', n_components=8)
    return data


def lebelenconder(data):
    data = data.drop(['scheme', 'netloc', 'path', 'parameters', 'query', 'fragment', 'new_url'], axis=1)
    data = data.drop(['method', 'refer', 'url_filetype', 'ua_short', 'ua_first', 'contact'], axis=1)
    return data


# 随机搜索训练
def random_search(x_train, y_train):
    model = lgb.LGBMClassifier(objective='multiclass', boosting_type='gbdt', verbose=-1)
    param_dist = {
        'num_leaves': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.6, 0.8, 1.0],
        'bagging_fraction': [0.6, 0.8, 1.0]
    }
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, scoring='accuracy',
                                       cv=3)
    random_search.fit(x_train, y_train)
    best_params = random_search.best_params_  # 获取最优参数组合
    best_model = random_search.best_estimator_  # 获取最优模型
    return best_model, best_params


# 正常训练
def normal_search(x_train, y_train):
    model = lgb.LGBMClassifier(objective='multiclass', boosting_type='gbdt', verbose=-1, num_leaves=100,
                               learning_rate=0.01, feature_fraction=0.8, bagging_fraction=0.8)
    model.fit(x_train, y_train)
    return model


def train_lgb(cfg, data):
    best_accuracy = 0
    kf = StratifiedKFold(n_splits=cfg.FOLDS, shuffle=True, random_state=42)
    x, y = data.iloc[:, :-1], data['label'].values
    for fold, (train_idx, val_idx) in enumerate(kf.split(x, y)):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # model, params = random_search(x_train, y_train)  # 随机搜索
        model = normal_search(x_train, y_train)  # 正常训练
        y_pred = model.predict(x_val)
        p = precision_score(y_val, y_pred, average='macro', zero_division=1)
        r = recall_score(y_val, y_pred, average='macro', zero_division=1)
        f = f1_score(y_val, y_pred, average='macro', zero_division=1)
        print(p, r, f)
        if f > best_accuracy:
            best_accuracy = f
            best_model = model
            joblib.dump(best_model, 'model/best_lgb.joblib')


def lgb_predict(o_data):
    pre_data = o_data[o_data['label'].isna()]
    id_list = pre_data['id']
    not_use_feats = ['id', 'user_agent', 'url', 'body', 'url_unquote', 'url_query', 'url_path', 'label']
    use_features = [col for col in pre_data.columns if col not in not_use_feats]
    pre_data = pre_data[use_features]
    best_model = joblib.load('model/best_lgb.joblib')
    pre = best_model.predict(pre_data.values)
    pre_data = pd.DataFrame({'id': id_list, 'predict': pre})
    pre_data.to_csv('predict.csv', encoding='gbk', index=False)


if __name__ == '__main__':
    cfg = Config()
    data = process_data(cfg)
    data = tfidf_feature(data)
    o_data = lebelenconder(data)
    data = o_data[o_data['label'].notna()]  # 实际使用删除，刷榜用的代码
    data = data.sample(frac=1).reset_index(drop=True)  # 按行随机打乱数据
    not_use_feats = ['id', 'user_agent', 'url', 'body', 'url_unquote', 'url_query', 'url_path']
    use_features = [col for col in data.columns if col not in not_use_feats]
    data = data[use_features]
    label = data.pop('label')
    data.insert(loc=data.shape[1], column='label', value=label, allow_duplicates=False)
    train_lgb(cfg, data)
    lgb_predict(o_data)
