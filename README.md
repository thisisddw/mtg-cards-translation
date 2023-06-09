# mtg-cards-translation

## 5.11

重构数据集。

- 两个系列bro和one被移出训练数据，用于测试整体效果
- rule-text原先的两个版本plain和default被移到outdated目录下，新增两个版本v1和v2

测试新想法。

- 用"<>"标记卡牌名称，训练模型原样输出尖括号中间的内容，不对它们做翻译
- 模型可以学到<>规则，但是整体翻译质量看起来下降了。可能是因为把卡名按字母拆开导致序列太长了
- 接下来应该尝试用标签（比如"<1>"）代替卡名，最后再替换

## 5.13

- 数据集新增rule-text/cnd，用于训练卡牌名称检测模型
- 新增models/card_name_detector模块，定义了从规则文字的单个句子中检测卡牌名的模型，基于双向RNN
- 简单训练了一个名称检测模型，可以通过models.card_name_detector.definition.TrainedDetector加载

## 5.21

utils/translate.py增加sentencize和CardTranslator，用于翻译包含多个句子的一段文本。

## 5.26

utils添加了calculate_testset_bleu，用于计算模型在测试系列上的bleu score。

## 5.27

- 新增数据集rule-text/v2.1，用\<id\>标签取代牌名和一部分关键字
- test_bench上训练模型model4-rule-v2.1并测试，效果不理想，模型没有很好地学会\<id\>标签的规则
- 改进方向：
    - 增强数据
    - 尝试transformer模型

## 6.9

- 添加models.model6