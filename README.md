# daguan_2019
2019达观杯实体识别

[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/torch-1.1.0-brightgreen.svg)](https://pypi.org/project/torch/1.1.0)
[![](https://img.shields.io/badge/pytorch--transformers-1.1.0-brightgreen.svg)](https://pypi.org/project/pytorch-transformers/1.1.0)
[![](https://img.shields.io/badge/tensorflow--gpu-1.12.0-brightgreen.svg)](https://pypi.org/project/tensorflow-gpu/1.12.0)
[![](https://img.shields.io/badge/keras-2.2.4-brightgreen.svg)](https://pypi.org/project/keras/2.2.4)
[![](https://img.shields.io/badge/numpy-1.16.2-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.16.2)
[![](https://img.shields.io/badge/gensim-3.4.0-brightgreen.svg)](https://pypi.python.org/pypi/gensim/3.4.0)

## **项目简介**
2019达观杯，实体识别代码分享。<br>
该比赛将文本全部为编码，所以需要自行做预训练。由于当时在做其他比赛和准备参展人工智能大会，所以大致用了3天初步训练了word2vec和BERT，并没有深入研究，代码仅供参考。<br>

## **思路简介**
比赛本身是一个相对简单的实体识别任务，难点主要在预训练上。所以该比赛可以尝试以下方案：<br><br>
1.仅使用训练数据，用embedding的方式随网络一起训练；<br><br>
2.根据给的百万语料，自行训练word2vec，做成权重矩阵载入embedding层，此处我使用gensim来训练；<br><br>
3.根据给的百万语料，自行训练BERT，我使用tf官方bert脚本来训练；<br><br>

## **训练word2vec**
1.运行```python make_corpus.py```，构建语料（里面的字典可以不构建，看到底需不需要全部词语词向量），词频从高到低，去掉词频为1的；<br><br>
2.运行```python train_word2vec.py --size 300 --sg 1 --workers 6 ```，用gensim训练word2vec，自行参考gensim函数说明修改epoch等参数，耗时较长建议后台运行，大概30-60分钟不等，取决于你的参数；<br><br>
3.运行```python make_weight.py --size 300 --sg 1 &```，结合词典（如果做了的话）和gensim模型，生成embedding权重；<br><br>

## **训练BERT**
本次比赛的重头戏，我也是第一次自己训练，还是觉得有点成就感的。采用谷歌官方脚本:<https://github.com/google-research/bert>，<br><br>
1.运行```python make_corpus.py```，构建语料（里面的字典可以不构建，看到底需不需要全部词语），把训练集、测试集、语料集都合并进texts_all.txt中，空格分隔；<br><br>
2.运行```make_vocab.py```，用到了之前生成字典，当然你也可以不用，主要看你的vocab到底需不需要全部词语；<br><br>
3.运行```python create_pretraining_data.py --input_file=./data/texts_all.txt --output_file=./data/texts_all_160_24_4_1.tfrecord --vocab_file=./data/vocab.txt --do_lower_case=True --max_seq_length=160 --max_predictions_per_seq=24 --masked_lm_prob=0.15 --random_seed=4 --dupe_factor=1```，构建语料，耗时较长建议后台运行；<br><br>
4.运行```CUDA_VISIBLE_DEVICES=0 python run_pretraining.py --input_file=./data/texts_all_160_24_1_1.tfrecord --output_dir=./models/texts_all_160_24_1_1/ --do_train=True --do_eval=True --bert_config_file=./data/bert_config.json --train_batch_size=32 --eval_batch_size=8 --max_seq_length=160 --max_predictions_per_seq=24 --num_train_steps=100000 --num_warmup_steps=10000 --learning_rate=5e-5 --save_checkpoints_steps 2000 --iterations_per_loop 2000```，耗时较长建议后台运行，1080ti需要若干天（取决于你的参数），不支持分布式，原因不详，所以我用了4块卡分别训练了4个不同种子的模型；<br><br>
5.运行```python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path ./try/model.ckpt-28000 --bert_config_file ./try/bert_config.json --pytorch_dump_path ./try/pytorch_model.bin```，转为pytorch模型，如果你是tf选手可以忽略此步；<br><br>

## **训练NER模型**
这个没啥含金量，无非是crf、指针网络等思路，各种比赛都差不多。<br><br>
1.运行```python train.py --cuda 3 --pretrain texts_all_160_24_4_1_100000 --num_layers 3 --hidden_dim 768 --loss_weight 3 --epochs 25 --k 0.865 --size 768```（3号卡、预训练名为texts_all_160_24_4_1_100000、3层、lstm768、损失权重比3、25个epoch、模型保存阈值0.865、bert模型输出维度）；<br><br>
2.运行```python submit_test.py```，选择模型输出结果的名称，生成提交结果；<br><br>

## **模型评估**
只花了一点点的时间做预训练，多训练几轮的效果会有显著提升；<br>

预训练Model | 线上得分（单模）| 线上得分（融合）
---------- | -------- | --------
无 | 0.84 | 0.86
word2vec | 0.86 | 0.88
bert | 0.88 | 0.90

## **其他说明**
因为最近讨论bert的一些细节，才记起来有这个比赛。当初只是练手训练bert，而且做的时候离结束就剩一周时间，所以没怎么深挖，如有问题请自行debug。整体应该没啥大问题，因为我线上提交过4次，对主办方提供比赛数据表示感谢。
