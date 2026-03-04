# %% [markdown]
# # 文本预处理
# 
# ## 加载数据集
# 
# 采用经典语料 [时光机器 (The Time Machine)](https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt)，约 30,000 词。
# 
# 首先按行读取，忽略标点符号和大小写，返回一个字符串列表
# %%
from enum import Enum
from typing import Callable

def remove_non_alpha_and_lower(stirng: str) -> str:
    # 清洗为只保留字母并转为小写
    import re
    return re.sub('[^a-zA-Z]+', ' ', stirng.lower()).strip()

def get_timemachine_lines(line_processor: Callable[[str], str] = remove_non_alpha_and_lower) -> list[str]:
    import os
    url = r'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    file_path = 'timemachine.txt'

    if not os.path.exists(file_path):
        import requests
        print(f'Downloading {url} to {file_path}')
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f'Error downloading {url} to {file_path}, Error code: {response.status_code}')
    with open(file_path, 'r') as f:
        return [line_processor(line) for line in f]

if __name__ == '__main__':
    lines = get_timemachine_lines(line_processor=remove_non_alpha_and_lower)
    print(*lines, sep='\n')
# %% [markdown]
# ## 词元化
# %%
from typing import Literal

def tokenize(string: str, token_type: Literal['word', 'char']) -> list[str]:
    match token_type:
        case 'word':
            return string.split()
        case 'char':
            return list(string)

if __name__ == '__main__':
    tokens = [tokenize(line, token_type='word') for line in get_timemachine_lines()]
    for i in range(10):
        print(tokens[i])
# %% [markdown]
# 为了一些特定内容、任务和结构，引入一些特殊词元，如：
# 
# - <UNK>：未被纳入词表的未知词元 (unknown token)；
# 
# - <PAD>：用于填充序列长度的标记 (padding token)；
# 
# - <SOS>：序列的开始标记 (start of sequence)；
# 
# - <EOS>：序列的结束标记 (end of sequence)；
# 
# - <SEP>：段落的分隔标记 (separator token)；
# 
# - <MASK>：掩盖以用于预测的标记 (mask token)。
# %%
from enum import Enum
class ST(str, Enum):
    """特殊词元 (Special Token)"""

    UNK = '<UNK>'
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    SEP = '<SEP>'
    MASK = '<MASK>'

    @property
    def description(self):
        describe = {
            ST.UNK: '未被纳入词表的未知词元 (unknown token)',
            ST.PAD: '用于填充序列长度的标记 (padding token)',
            ST.SOS: '序列的开始标记 (start of sequence)',
            ST.EOS: '序列的结束标记 (end of sequence)',
            ST.SEP: '段落的分隔标记 (separator token)',
            ST.MASK: '掩盖以用于预测的标记 (mask token)',
        }
        return describe[self.value]
# %% [markdown]
# ## 构建词表
# 
# 模型的训练与推理只能使用数值完成，因此需要将词元用数值唯一标识。
# %%
from collections import Counter
from typing import Optional, Iterable

class Vocabulary:
    def __init__(self,
                 flat_tokens: list[str],
                 special_tokens: Iterable[ST],
                 min_freq: int  = 1):
        """
        :param flat_tokens: 被展平后的词元列表
        :param min_freq: 纳入词表的最小词频
        :param special_tokens: 由特殊词元组成的可迭代对象
        """
        self.__unk_token = ST.UNK
        # 先放入特殊词元
        tokens: list[str] = list(dict.fromkeys([self.__unk_token] + [i for i in special_tokens]))
        # 按最小词频过滤
        self.__valid_token_freqs = {token: freq for token, freq in Counter(flat_tokens).items() if freq >= min_freq}
        # 加入过滤后的有效词元
        tokens.extend([
            token for token, freq in
            sorted(self.__valid_token_freqs.items(), key=lambda pair: pair[1], reverse=True)
        ])

        # 双向映射表
        self.__token_to_index = {token: index for index, token in enumerate(tokens)}
        self.__index_to_token = {index: token for token, index in self.__token_to_index.items()}

    def __len__(self):
        return len(self.__token_to_index)

    def get_index(self, token: str) -> int:
        return self.__token_to_index.get(
            token,
            self.__token_to_index[self.__unk_token]
        )

    def get_token(self, index: int) -> str:
        return self.__index_to_token.get(
            index,
            self.__unk_token
        )

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.get_index(token) for token in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.get_token(index) for index in indices]

    def __repr__(self):
        return f'Vocabulary({self.__token_to_index})'

    @property
    def vocabulary(self) -> dict[str, int]:
        return self.__token_to_index

    @property
    def valid_token_freqs(self) -> dict[str, int]:
        return self.__valid_token_freqs
# %%
def get_vocab_corpus_from_timemachine(
        token_type: Literal['word', 'char'],
        special_tokens: Iterable[ST] = (ST.UNK, ),
        max_token_num: Optional[int] = None,
) -> (Vocabulary, list[int]):
    tokenized_lines = [tokenize(line, token_type=token_type) for line in get_timemachine_lines()]
    flat_tokens = [token for line in tokenized_lines for token in line]
    vocab_instance = Vocabulary(flat_tokens, special_tokens=special_tokens)
    list_token_indices = [vocab_instance.get_index(token) for token in flat_tokens]
    return vocab_instance, list_token_indices[:max_token_num] if max_token_num else list_token_indices
# %%
if __name__ == '__main__':
    vocab, corpus = get_vocab_corpus_from_timemachine(token_type='word')

    print(f'词元索引列表大小为：{len(corpus)}')
    print(f'词汇表的大小为：{len(vocab)}')
    print(f'词元频率字典（前 10 个）：{list(vocab.valid_token_freqs.items())[:10]}')

    for i in range(8):
        print(f'索引值为 {i} 的词元为：{vocab.get_token(i)!r}')