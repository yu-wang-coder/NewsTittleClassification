## 基于THUCNews数据集的中文文本分类

## 介绍

使用清华大学自然语言处理实验室中[THUCNews](http://thuctc.thunlp.org/)的部分20万条新闻标题，使用Transformer作为基础框架，以“字”为单位进行分词，使用搜狗新闻中文词向量作为模型的词嵌入层，训练了一个短文本分类器。

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## 代码介绍
```
│  README.md
│  run.py                                   # 直接运行可以重新训练一个模型
│  text_classification.py                   # 已经实现好的文本分类器，可直接使用
│  train_eval.py                            # 训练过程代码
│  utils.py                                 # 数据预处理代码
│  
├─models
│  │  Transformer.py                        # 模型框架代码
│          
├─THUCNews                                  # 数据集、字典和词向量
│  ├─data
│  │      class.txt                         # 类别表
│  │      dev.txt                           # 验证集
│  │      embedding_SougouNews.npz          # 搜狗新闻预训练词向量
│  │      embedding_Tencent.npz             # 腾讯预训练词向量
│  │      test.txt                          # 测试集
│  │      train.txt                         # 训练集
│  │      vocab.pkl                         # 字典
│  │      
│  ├─log                                    # 训练日志 
│  │  └─Transformer
│  │      ├─06-17_21.06
│  │      │      events.out.tfevents.1718629576.LAPTOP-RIEDPD0U
│  │      │      
│  │      └─06-17_23.27
│  │              events.out.tfevents.1718638035.LAPTOP-RIEDPD0U
│  │              
│  └─saved_dict                             # 训练好的模型参数
│          Transformer.ckpt
```

## 代码运行方法

可以自己重新训练一个基于Transformer的文本分类模型
```sh
python run.py
```
也可以直接使用提供的训练好的模型进行文本分类
```sh
python text_classification.py
```

## 代码参考
[https://github.com/649453932/Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)