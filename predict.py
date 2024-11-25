from dataset import process_data
from Config import Config
import torch
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def add_tfidata_feats(data, col, n_components=16):
    text = list(data[col].values)
    tf = joblib.load('model/{}_tfidf.joblib'.format(col))
    X = tf.transform(text)
    svd = joblib.load('model/{}_svd.joblib'.format(col))
    X_svd = svd.transform(X)
    for i in range(n_components):
        data[f'{col}_tfidata_{i}'] = X_svd[:, i]
    return data


def tfidf_feature(data):
    data = add_tfidata_feats(data, 'url_unquote', n_components=16)
    data = add_tfidata_feats(data, 'user_agent', n_components=16)
    data = add_tfidata_feats(data, 'body', n_components=32)
    data['contact'] = data['method'] + ' ' + data['refer'] + ' ' + data['url_filetype'] + ' ' + data['ua_short'] + ' ' + \
                      data['ua_first']
    data = add_tfidata_feats(data, 'contact', n_components=16)
    return data


def lebelenconder(data):

    # for col in ['method', 'refer', 'url_filetype', 'ua_short', 'ua_first']:
    #     le = LabelEncoder()
    #     le.fit(data[col])
    #     joblib.dump(le, 'model/{}_labelencoder.joblib'.format(col))
    #     data[col] = le.transform(data[col])
    data =data.drop(['method', 'refer', 'url_filetype', 'ua_short', 'ua_first','contact'],axis=1)
    return data


if __name__ == '__main__':
    cfg = Config()
    cfg.root_path = 'data/test'
    data = process_data(cfg)
    data = tfidf_feature(data)
    data = lebelenconder(data)
    not_use_feats = ['id', 'user_agent', 'url', 'body', 'url_unquote', 'url_query', 'url_path']
    use_features = [col for col in data.columns if col not in not_use_feats]
    data = data[use_features]
    x = torch.tensor(data[use_features].values).to('cuda')
    model = torch.load('model/model.pt',weights_only=False)
    x = x.reshape(x.shape[0], 1, -1)
    pre = model(x.float())
    pre = pre.reshape(pre.shape[0], -1)
    _, pre = torch.max(pre, 1)
    pre = pre.detach().cpu().tolist()
    predict_data = pd.read_csv('data/test/test.csv')
    predict_data = pd.DataFrame({'id':predict_data['id'],'predict':pre})
    # predict_data['predict'] = pre
    print(predict_data.shape)
    print(len(pre))
    predict_data.to_csv('predict.csv', encoding='gbk', index=False)
