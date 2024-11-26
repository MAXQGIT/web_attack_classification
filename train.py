from dataset import process_data
from Config import Config
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from self_model import LSTM_MODEL
from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


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
    # for i in range(n_components):
    #     data[f'{col}_tfidata_{i}'] = X_svd[:, i]
    columns = [f'{col}_tfidata_{i}' for i in range(n_components)]
    data_svd = pd.DataFrame(X_svd, columns=columns)
    data = pd.concat([data, data_svd], axis=1)
    return data


def tfidf_feature(data):
    data = add_tfidata_feats(data, 'url_unquote', n_components=16)
    data = add_tfidata_feats(data, 'user_agent', n_components=16)
    data = add_tfidata_feats(data, 'body', n_components=32)
    # data = add_tfidata_feats(data, 'scheme', n_components=5)
    # data = add_tfidata_feats(data, 'netloc', n_components=5)
    # data = add_tfidata_feats(data, 'path', n_components=5)
    # data = add_tfidata_feats(data, 'parameters', n_components=5)
    # data = add_tfidata_feats(data, 'query', n_components=5)
    # data = add_tfidata_feats(data, 'fragment', n_components=5)
    data['contact'] = data['method'] + ' ' + data['refer'] + ' ' + data['url_filetype'] + ' ' + data['ua_short'] + ' ' + \
                      data['ua_first']
    data = add_tfidata_feats(data, 'contact', n_components=32)
    data['new_url'] = data['scheme'] + ' ' + data['netloc'] + ' ' + data['path'] + ' ' + data['parameters'] + ' ' + \
                      data['query'] + ' ' + data['fragment']
    data = add_tfidata_feats(data, 'new_url', n_components=32)
    return data


def lebelenconder(data):
    # for col in ['method', 'refer', 'url_filetype', 'ua_short', 'ua_first']:
    #     le = LabelEncoder()
    #     le.fit(data[col])
    #     joblib.dump(le, 'model/{}_labelencoder.joblib'.format(col))
    #     data[col] = le.transform(data[col])
    data = data.drop(['scheme', 'netloc', 'path', 'parameters', 'query', 'fragment', 'new_url'], axis=1)
    data = data.drop(['method', 'refer', 'url_filetype', 'ua_short', 'ua_first', 'contact'], axis=1)
    data = data[data['label'].notna()] #实际使用删除，刷榜用的代码
    return data


# def sample_data(df):
#     max_count = df['label'].value_counts().max()
#     df_oversampled = pd.DataFrame()
#     for label in df['label'].unique():
#         df_label = df[df['label'] == label]
#         oversample_count = max_count - len(df_label)
#         df_label_oversampled = df_label.sample(oversample_count, replace=True)
#         df_oversampled = pd.concat([df_oversampled, df_label, df_label_oversampled])
#     return df_oversampled

# def sample_data(data):
#     y =data[['label']]
#     x =data.drop(['label'], axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
#     df_resampled = pd.DataFrame(X_resampled, columns=x.columns)  # 用原来的列名
#     df_resampled['label'] = y_resampled
#     return df_resampled


def tensor_data(cfg, data):
    train_len = int(cfg.train_radio * len(data))
    train_data = data.iloc[:train_len, :]
    val_data = data.iloc[train_len:, :]
    x_train, y_train = torch.tensor(train_data.iloc[:, :-1].values), torch.tensor(train_data.iloc[:, -1].values)
    x_val, y_val = torch.tensor(val_data.iloc[:, :-1].values), torch.tensor(val_data.iloc[:, -1].values)
    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    train_data = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_data = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)
    return train_data, val_data


def test(cfg, loss_fn, model, test_data):
    train_loss_list, recall_score_list, precision_score_list, f1_score_list = [], [], [], []
    for x, y in test_data:
        x, y = x.to(cfg.device), y.to(cfg.device)
        pre = model(x.float())
        pre = pre.reshape(pre.shape[0], -1)
        loss = loss_fn(pre, y.to(torch.int64))
        train_loss_list.append(loss.item())
        _, pre = torch.max(pre, 1)
        y = y.detach().cpu().tolist()
        pre = pre.detach().cpu().tolist()
        precision_score_list.append(precision_score(y, pre, average='macro', zero_division=1))
        recall_score_list.append(recall_score(y, pre, average='macro', zero_division=1))
        f1_score_list.append(f1_score(y, pre, average='macro', zero_division=1))
    precision_score1 = sum(precision_score_list) / len(precision_score_list)
    recall_score1 = sum(recall_score_list) / len(recall_score_list)
    f1_score1 = sum(f1_score_list) / len(f1_score_list)
    train_loss = sum(train_loss_list) / len(train_data)
    return train_loss, precision_score1, recall_score1, f1_score1


if __name__ == '__main__':
    cfg = Config()
    data = process_data(cfg)
    data = tfidf_feature(data)
    data = lebelenconder(data)
    data = data.sample(frac=1).reset_index(drop=True)  # 按行随机打乱数据
    not_use_feats = ['id', 'user_agent', 'url', 'body', 'url_unquote', 'url_query', 'url_path']
    use_features = [col for col in data.columns if col not in not_use_feats]
    data = data[use_features]
    # data = sample_data(data)  # 对数据进行样本均衡处理
    label = data.pop('label')
    data.insert(loc=data.shape[1], column='label', value=label, allow_duplicates=False)
    train_data, val_data = tensor_data(cfg, data)
    cfg.input_dim = data.shape[1] - 1
    model = LSTM_MODEL(cfg).to(cfg.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(cfg.epochs):
        model.train()
        train_loss_list, recall_score_list, precision_score_list, f1_score_list = [], [], [], []
        for x, y in train_data:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pre = model(x.float())
            pre = pre.reshape(pre.shape[0], -1)
            optimizer.zero_grad()
            loss = loss_fn(pre, y.to(torch.int64))
            train_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            _, pre = torch.max(pre, 1)
            y = y.detach().cpu().tolist()
            pre = pre.detach().cpu().tolist()
            precision_score_list.append(precision_score(y, pre, average='macro', zero_division=1))
            recall_score_list.append(recall_score(y, pre, average='macro', zero_division=1))
            f1_score_list.append(f1_score(y, pre, average='macro', zero_division=1))
        scheduler.step()
        precision_score1 = sum(precision_score_list) / len(precision_score_list)
        recall_score1 = sum(recall_score_list) / len(recall_score_list)
        f1_score1 = sum(f1_score_list) / len(f1_score_list)
        train_loss = sum(train_loss_list) / len(train_data)
        torch.save(model, 'model/model.pt')
        model.eval()
        test_loss, test_precision_score1, test_recall_score1, test_f1_score1 = test(cfg, loss_fn, model, val_data)
        print(
            'epoch:{} train_loss:{:.2f} train_precision:{:.2f} train_recall:{:.2f} train_f1:{:.2f} test_loss:{:.2f} test_precision:{:.2f} test_recall:{:.2f} test_f1:{:.2f}'.format(
                epoch, train_loss, precision_score1, recall_score1, f1_score1, test_loss, test_precision_score1,
                test_recall_score1, test_f1_score1))
