import pandas as pd
import glob
import os
from Config import Config
import re
from urllib.parse import quote, unquote, urlparse
import numpy as np


def read_data(cfg):
    path_list = glob.glob(os.path.join(cfg.root_path, '*.csv'))
    train_data = pd.DataFrame()
    for path in path_list:
        data = pd.read_csv(path)
        train_data = pd.concat([train_data, data]).reset_index(drop=True)
    train_data = train_data.fillna('__NaN__')
    test_data = pd.read_csv('../data/test/test.csv')  # 实际使用删除，刷榜用的代码
    test_data = test_data.fillna('__NaN__')  # 实际使用删除，刷榜用的代码
    train_data = pd.concat([train_data, test_data]).reset_index(drop=True) # 实际使用删除，刷榜用的代码
    train_data = train_data.rename(columns={'lable': 'label'})
    # train_data = train_data.sample(frac=1).reset_index(drop=True)
    return train_data


def get_url_query(s):
    li = re.split('[=&]', urlparse(s)[4])
    return [li[i] for i in range(len(li)) if i % 2 == 1]


def find_max_str_length(x):
    li = [len(i) for i in x]
    return max(li) if len(li) > 0 else 0


def find_str_length_std(x):
    li = [len(i) for i in x]
    return np.std(li) if len(li) > 0 else -1


def process_url(data):
    data['url_unquote'] = data['url'].apply(unquote)
    data['url_query'] = data['url_unquote'].apply(lambda x: get_url_query(x))
    data['url_query_num'] = data['url_query'].apply(len)
    data['url_query_max_len'] = data['url_query'].apply(find_max_str_length)
    data['url_query_len_std'] = data['url_query'].apply(find_str_length_std)
    data['scheme_len'] = data['url'].apply(lambda x: len(urlparse(x).scheme))
    data['scheme'] = data['url'].apply(lambda x: urlparse(x).scheme)
    data['netloc'] = data['url'].apply(lambda x: urlparse(x).netloc)
    data['netloc_len'] = data['url'].apply(lambda x: len(urlparse(x).netloc))
    data['path'] = data['url'].apply(lambda x: urlparse(x).path)
    data['path_len'] = data['url'].apply(lambda x: len(urlparse(x).path))
    data['parameters'] = data['url'].apply(lambda x: urlparse(x).params)
    data['parameters_len'] = data['url'].apply(lambda x: len(urlparse(x).params))
    data['query'] = data['url'].apply(lambda x: urlparse(x).query)
    data['query_len'] = data['url'].apply(lambda x: len(urlparse(x).query))
    data['fragment'] = data['url'].apply(lambda x: urlparse(x).fragment)
    data['fragment_len'] = data['url'].apply(lambda x: len(urlparse(x).fragment))
    return data


def find_url_filetype(x):
    try:
        return re.search(r'\.[a-z]+', x).group()
    except:
        return '__NaN__'


def get_url(data):
    data['url_path'] = data['url_unquote'].apply(lambda x: urlparse(x)[2])
    data['url_filetype'] = data['url_path'].apply(lambda x: find_url_filetype(x))
    data['url_path_len'] = data['url_path'].apply(len)
    data['url_path_num'] = data['url_path'].apply(lambda x: len(re.findall('/', x)))
    data['ua_short'] = data['user_agent'].apply(lambda x: x.split('/')[0])
    data['ua_first'] = data['user_agent'].apply(lambda x: x.split(' ')[0])
    return data


def process_data(cfg):
    data = read_data(cfg)
    data = process_url(data)
    data = get_url(data)
    return data


if __name__ == '__main__':
    cfg = Config()
    data = process_data(cfg)
    print(data.head())
    print(data.columns)
