from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import os


class RuleText:
    versions = {'default', 'plain'}

    @classmethod
    def splits(cls, fields, version='default', train='train.json', validation='valid.json', test='test.json'):
        """
        Load data from ./data/rule-text/$version

        Each example has two fields .src and .trg, corresponding to source language
        and target language.

        Versions:
            default: Sentence level alignment. Replace card name with '<cn>' in trg, while src remains untouched.
            plain: Sentence level alignment without further processing.

        Todo:   1. 去掉重复的短句
                2. 把整张牌的文本作为样本
                3. 把某些单词替换成<unk>，或许可以改进低频词翻译错误的问题 （鼓励模型把少见的词翻译成<unk>）
        """
        path = os.path.dirname(os.path.abspath(__file__)) + '/data/rule-text/' + version + '/'

        return TabularDataset.splits(path=path,
                                     train=train,
                                     validation=validation,
                                     test=test,
                                     format='json',
                                     fields=fields)
class CardName:
    versions={'swamp'}
    """
        Load data from ./data/card-name/$version

        Each example has two fields .src and .trg, corresponding to source language
        and target language.

        Versions:
            swamp: nothing to do

        """
    @classmethod
    def splits(cls, fields, version='swamp', train='train.json', validation='valid.json', test='test.json'):
        path = os.path.dirname(os.path.abspath(__file__)) + '/data/card-name/' + version + '/'
        return TabularDataset.splits(path=path,
                                     train=train,
                                     validation=validation,
                                     test=test,
                                     format='json',
                                     fields=fields)
