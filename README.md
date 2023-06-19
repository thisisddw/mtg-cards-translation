# mtg-cards-translation
## 目录结构
```
.
├─dataset 
│  └─data   # 每一个子文件夹代表一个数据集
│      ├─card-name      # 存放卡牌名数据集
│      │  └─swamp
│      ├─rule-text      # 存放规则文字数据集
│      │  ├─cnd         # 训练名称检测模型的数据集
│      │  ├─out-dated 
│      │  │  ├─default  # 中文句子中的卡牌名被替换为标记 <cn>
│      │  │  └─plain    # 原始版本
│      │  ├─v1          # 中文句子中的卡牌名被替换为标记 <cn>
│      │  ├─v2          # 英文句子中的卡牌名被加上尖括号标记，中文句子中出现的卡牌名被替换为英文名
│      │  ├─v2.1        # 一些中英文卡牌名被替换为 <id> 标签
│      │  └─v2.2        # 在 v2.1 的基础上做了更多替换，并且保留了未做替换的版本
│      └─test-sets
├─demo      # 演示网页
├─models
│  ├─card_name_detector     # 名称检测模型
│  ├─hybrid                 # 混合模型
│  ├─model4                 # RNNSearch模型
│  ├─model6                 # Transformer (可学习位置编码)  
│  └─model6_1               # Transformer (固定位置编码)
├─result    # 存放模型参数
└─utils
```
## 日志

### 5.11

重构数据集。

- 两个系列bro和one被移出训练数据，用于测试整体效果
- rule-text原先的两个版本plain和default被移到outdated目录下，新增两个版本v1和v2

测试新想法。

- 用"<>"标记卡牌名称，训练模型原样输出尖括号中间的内容，不对它们做翻译
- 模型可以学到<>规则，但是整体翻译质量看起来下降了。可能是因为把卡名按字母拆开导致序列太长了
- 接下来应该尝试用标签（比如"<1>"）代替卡名，最后再替换

### 5.13

- 数据集新增rule-text/cnd，用于训练卡牌名称检测模型
- 新增models/card_name_detector模块，定义了从规则文字的单个句子中检测卡牌名的模型，基于双向RNN
- 简单训练了一个名称检测模型，可以通过models.card_name_detector.definition.TrainedDetector加载

### 5.21

utils/translate.py增加sentencize和CardTranslator，用于翻译包含多个句子的一段文本。

### 5.26

utils添加了calculate_testset_bleu，用于计算模型在测试系列上的bleu score。

### 5.27

- 新增数据集rule-text/v2.1，用\<id\>标签取代牌名和一部分关键字
- test_bench上训练模型model4-rule-v2.1并测试，效果不理想，模型没有很好地学会\<id\>标签的规则
- 改进方向：
    - 增强数据
    - 尝试transformer模型

### 6.9

- 添加models.model6，基于transformer
- 新增数据集rule-text/v2.2, 将替换关键字和未替换关键字的版本共同加入数据集，减少了因添加标签导致的训练集语料损失，并解决了训练集里标签位置单一的问题
- 在model4上测试了新的数据集，效果有了提升，且对标签替换规则掌握的更好了
- 进一步工作
    - 尝试model6在新数据集上的效果
  
### 6.10
- model6在测试中良好的掌握了单词的含义，但存在语序混乱，颠三倒四的情况
- 改进想法：尝试其他位置编码方法，看看模型是否能更好的学习到语言的顺序
- 尝试改用固定编码方法，效果很好
- 尝试使用混合模型，效果一般

### 6.11
- 编写报告，进行五种模型全面对比
- 保存所有训练好的模型，增加一个用来进行对比测试的test文件
  
