import pickle
import tqdm
from collections import Counter

# 定义 TorchVocab 类，用于创建和管理词汇表，将文本 token 转换为数字索引
class TorchVocab(object):
    """定义一个词汇表对象，用于将字段数值化。
    属性：
        freqs: 一个 collections.Counter 对象，存储数据中 token 的频率。
        stoi: 一个 collections.defaultdict，映射 token 字符串到数字标识符。
        itos: 一个 token 字符串列表，按数字标识符索引。
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """从 collections.Counter 创建 Vocab 对象。
        参数：
            counter: 包含数据中每个值的频率的 collections.Counter 对象。
            max_size: 词汇表的最大大小，None 表示无限制。默认：None。
            min_freq: 包含在词汇表中的 token 最小频率，小于 1 的值设为 1。默认：1。
            specials: 特殊 token 列表（例如填充或结束符），会添加到词汇表开头。默认：['<pad>']。
            vectors: 预训练词向量或自定义词向量。
            unk_init: 默认初始化未知词向量为零向量，可以是任意函数，接受 Tensor 并返回相同大小的 Tensor。
            vectors_cache: 缓存词向量的目录。默认：'.vector_cache'。
        """
        # 保存 token 频率计数器
        self.freqs = counter
        # 复制计数器以避免修改原始数据
        counter = counter.copy()
        # 确保 min_freq 至少为 1
        min_freq = max(min_freq, 1)

        # 初始化 itos（index to string）列表，包含特殊 token
        self.itos = list(specials)
        # 在构建词汇表时，不计算特殊 token 的频率
        for tok in specials:
            del counter[tok]

        # 如果指定了 max_size，调整为包含特殊 token 后的总大小
        max_size = None if max_size is None else max_size + len(self.itos)

        # 按频率降序排序，若频率相同则按字母顺序排序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # 将符合条件的 token 添加到 itos 列表
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # 创建 stoi（string to index）字典，反向映射 itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        # 初始化词向量为 None
        self.vectors = None
        # 如果提供了词向量，加载它们
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            # 如果没有提供词向量，确保 unk_init 和 vectors_cache 为 None
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        # 定义相等比较，检查 freqs、stoi、itos 和 vectors 是否相同
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        # 返回词汇表的大小（itos 列表的长度）
        return len(self.itos)

    def vocab_rerank(self):
        # 重新生成 stoi 字典，根据当前 itos 重新编号
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        # 扩展词汇表，添加另一个词汇表中的 token
        # 如果 sort=True，则按字母顺序添加；否则按原顺序
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

# 定义 Vocab 类，继承自 TorchVocab，添加特定 token 的索引
class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        # 定义特殊 token 的索引
        self.pad_index = 0  # 填充 token 的索引
        self.unk_index = 1  # 未知 token 的索引
        self.eos_index = 2  # 句子结束 token 的索引
        self.sos_index = 3  # 句子开始 token 的索引
        self.mask_index = 4  # 掩码 token 的索引
        # 调用父类构造函数，传入特殊 token 列表
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
        # 将句子转换为索引序列（待实现）
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        # 将索引序列转换回句子（待实现）
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        # 从文件中加载词汇表对象
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        # 将词汇表对象保存到文件
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

# 定义 WordVocab 类，继承自 Vocab，专门处理文本文件构建词汇表
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        # 打印提示信息，表示开始构建词汇表
        print("Building Vocab")
        # 初始化 Counter 对象，用于统计 token 频率
        counter = Counter()
        # 遍历输入的文本数据，显示进度条
        for line in tqdm.tqdm(texts):
            # 如果输入是列表，直接使用；否则按空格分词
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            # 统计每个 token 的频率
            for word in words:
                counter[word] += 1
        # 调用父类构造函数，传入统计结果
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        # 将输入句子转换为索引序列
        # 如果输入是字符串，先分词
        if isinstance(sentence, str):
            sentence = sentence.split()

        # 将每个词转换为对应的索引，未知词使用 unk_index
        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        # 如果需要，添加句子结束 token
        if with_eos:
            seq += [self.eos_index]  # 索引 2
        # 如果需要，添加句子开始 token
        if with_sos:
            seq = [self.sos_index] + seq  # 索引 3

        # 保存原始序列长度
        origin_seq_len = len(seq)

        # 如果指定了序列长度，调整序列
        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            # 如果序列短于指定长度，用 pad_index 填充
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            # 如果序列长于指定长度，截断
            seq = seq[:seq_len]

        # 如果需要返回长度，返回序列和原始长度；否则只返回序列
        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        # 将索引序列转换回词列表
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx  # 对于超出词汇表大小的索引，返回占位符
                 for idx in seq
                 if not with_pad or idx != self.pad_index]  # 如果不包含填充 token，则跳过 pad_index

        # 如果 join=True，返回拼接的字符串；否则返回词列表
        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        # 从文件中加载 WordVocab 对象
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

# 定义 build 函数，用于从文本文件构建词汇表并保存
def build():
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需的输入文件路径参数
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    # 添加必需的输出文件路径参数
    parser.add_argument("-o", "--output_path", required=True, type=str)
    # 添加词汇表最大大小参数，默认 None
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    # 添加文件编码参数，默认 utf-8
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    # 添加最小频率参数，默认 1
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    # 解析命令行参数
    args = parser.parse_args()

    # 打开输入文件，读取文本数据
    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        # 使用文本数据构建 WordVocab 对象
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    # 打印词汇表大小
    print("VOCAB SIZE:", len(vocab))
    # 将词汇表保存到指定输出路径
    vocab.save_vocab(args.output_path)