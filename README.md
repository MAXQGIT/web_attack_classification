网络攻击分类算法是临时看到 https://www.datafountain.cn/competitions/1068 有个训练赛，感觉挺有意思就随便写了一个算法。

验证集合上准确率和召回率分别是0.94 0.94 感兴趣的可以参考一下。

本人完全不懂网络请求协议中东西，自己在网上找了一点请求的东西，写了特征提取算法，（这个特征提取肯定存在问题，但是我完全不懂网络请求中的东西，所以特征提取需要改动）

算法中涉及有简单的数据增强算法，实际测试后发现，不如不增强，因此将数据增强算法逻辑屏蔽了

数据获取链接 通过网盘分享的文件：网络攻击数据.zip
链接: https://pan.baidu.com/s/1DNVQ4Emj5k9d46vcrV7Ajw 提取码: ymqw

将数据解压成data文件夹，创建model文件夹

运行 python train.py

预测 python predict.py

希望给看到的人一点点帮助。