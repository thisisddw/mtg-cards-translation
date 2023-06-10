from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import os


class RuleText:
    versions = {'v1', 'v2', 'v2.1','v2.2', 'cnd', 'outdated/default', 'outdated/plain'}

    @classmethod
    def splits(cls, fields, version='v1', train='train.json', validation='valid.json', test='test.json'):
        """
        Load data from ./data/rule-text/$version

        Each example has two fields .src and .trg, corresponding to source language
        and target language.

        Recent update remove 2 sets from rule-text dataset for testing reason. Two previous versions were move
        into outdated directory.

        Versions:
            v1: Sentence level alignment. Replace card name with '<cn>' in trg, while src remains untouched.
            v2: Sentence level alignment. Wrap card name in src with pair of '<>', substitute card name in trg with <src-name>.
                Aiming at teaching the modal to leave string wraped in '<>' notations untouched.
            v2.1: Substitute card name and some keywords with '<id>', id is 0-9.
            v2.2: add more dictionary items during training
            cnd: Card name detection. Used to train a model to detect card names in rule text.
                Already tokenized, seperated with spaces.
            outdated/default: Sentence level alignment. Replace card name with '<cn>' in trg, while src remains untouched.
            outdated/plain: Sentence level alignment without further processing.

        Todo:   1. 去掉重复的短句
                2. 把整张牌的文本作为样本
                3. 把某些单词替换成<unk>，或许可以改进低频词翻译错误的问题 （鼓励模型把少见的词翻译成<unk>）
        """
        assert version in cls.versions
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

class TestSets:
    """
    Sets that not appear in training.
    Fields: key, src-name, trg-name, stc-rule, trg-rule, src-flavor, trg-flavor
    """

    versions = {'bro', 'one'}

    @classmethod
    def load(cls, fields, version='one'):
        """
        Load data from ./data/test-sets/$version

        Available versions: bro, one
        """
        path = os.path.dirname(os.path.abspath(__file__)) + '/data/test-sets/'

        return TabularDataset.splits(path=path,
                                     train=version,
                                     validation=None,
                                     test=None,
                                     format='json',
                                     fields=fields)[0]