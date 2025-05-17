from torch.utils.data import Dataset
# 导入 PyTorch 的 Dataset 类，用于自定义数据集

import tqdm
# 导入 tqdm 库，用于显示数据加载的进度条

import torch
# 导入 PyTorch 库，用于张量操作

import random
# 导入 random 模块，用于随机化操作

class BERTDataset(Dataset):
    # 定义 BERTDataset 类，继承 PyTorch 的 Dataset 类，用于 BERT 模型的数据预处理

    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        # 初始化方法，接收语料路径、词汇表、序列长度、编码格式、语料行数和是否加载到内存的参数
        self.vocab = vocab
        # 保存词汇表对象，包含词到索引的映射和特殊标记（如 SOS、EOS 等）
        self.seq_len = seq_len
        # 保存最大序列长度，用于截断或填充序列
        self.on_memory = on_memory
        # 布尔值，决定是否将整个数据集加载到内存中
        self.corpus_lines = corpus_lines
        # 语料行数，初始可能为 None，如果未指定则需要计算
        self.corpus_path = corpus_path
        # 语料文件路径
        self.encoding = encoding
        # 文件编码格式，默认为 utf-8

        with open(corpus_path, "r", encoding=encoding) as f:
            # 打开语料文件，准备读取数据
            if self.corpus_lines is None and not on_memory:
                # 如果未指定语料行数且不加载到内存，计算文件总行数
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    # 使用 tqdm 显示加载进度条，迭代文件每一行
                    self.corpus_lines += 1
                    # 累加行数，统计语料总行数

            if on_memory:
                # 如果选择加载到内存
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                # 读取文件每一行，去掉末尾换行符，按制表符分割，存储为列表
                self.corpus_lines = len(self.lines)
                # 更新语料行数为实际读取的行数

        if not on_memory:
            # 如果不加载到内存
            self.file = open(corpus_path, "r", encoding=encoding)
            # 打开语料文件，保持文件句柄用于后续按需读取
            self.random_file = open(corpus_path, "r", encoding=encoding)
            # 打开另一个文件句柄，用于随机读取行

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                # 随机跳过若干行，限制最大跳跃行数为 1000 或语料总行数
                self.random_file.__next__()
                # 前进文件指针到随机位置

    def __len__(self):
        # 定义数据集长度方法，返回数据集的总行数
        return self.corpus_lines
        # 返回语料的总行数

    def __getitem__(self, item):
        # 定义获取单个样本的方法，根据索引 item 返回处理后的数据
        t1, t2,s1,s2,is_next_label = self.random_sent(item)
        # 获取两个句子及其是否连续的标签（1 表示连续，0 表示不连续）
        t1_random, t1_label = self.random_word(t1)
        # 对第一个句子进行随机词替换（掩码或随机词），返回替换后的句子和标签
        t2_random, t2_label = self.random_word(t2)
        # 对第二个句子进行随机词替换，返回替换后的句子和标签

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # 注释说明：CLS 标记对应起始标记（SOS），SEP 标记对应结束标记（EOS）
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        # 在第一个句子前后添加 SOS 和 EOS 标记
        t2 = t2_random + [self.vocab.eos_index]
        # 在第二个句子后添加 EOS 标记
        s1 = [1.0] + s1 + [1.0]
        s2 = s2 + [1.0]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        # 为第一个句子的标签前后添加填充标记（pad_index）
        t2_label = t2_label + [self.vocab.pad_index]
        # 为第二个句子的标签后添加填充标记

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        # 创建段落标签：第一个句子标记为 1，第二个句子标记为 2，截断到 seq_len
        bert_input = (t1 + t2)[:self.seq_len]
        # 拼接两个句子，截断到最大序列长度
        bert_label = (t1_label + t2_label)[:self.seq_len]
        # 拼接两个句子的标签，截断到最大序列长度
        s_label = (s1 + s2)[:self.seq_len]
        s_padding = [1.0 for _ in range(self.seq_len - len(s_label))]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        # 计算需要填充的长度，创建填充标记列表
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding), s_label.extend(s_padding)
        # 对输入、标签和段落标签进行填充，使长度达到 seq_len
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label,
                  "s_label": s_label}
        # 创建输出字典，包含 BERT 输入、标签、段落标签和是否连续标签

        return {key: torch.tensor(value) for key, value in output.items()}
        # 将输出字典中的每个值转换为 PyTorch 张量并返回

    def random_word(self, sentence):
        # 定义随机词替换方法，模拟 BERT 的掩码语言模型（MLM）任务
        tokens = sentence
        # 将句子按空格分割为词列表
        output_label = []
        # 初始化标签列表，用于记录原始词的索引

        for i, token in enumerate(tokens):
            # 遍历每个词
            prob = random.random()
            # 生成 0 到 1 的随机数，用于决定是否替换该词
            if prob < 0.15:
                # 以 15% 的概率对词进行替换
                prob /= 0.15
                # 将概率归一化到 0-1 范围，用于后续子条件判断

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # 80% 的概率（15% * 80% = 12%）将词替换为掩码标记
                    tokens[i] = self.vocab.mask_index
                    # 将当前词替换为词汇表中的掩码标记索引

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # 10% 的概率（15% * 10% = 1.5%）将词替换为随机词
                    tokens[i] = random.randrange(len(self.vocab))
                    # 从词汇表中随机选择一个词的索引

                # 10% randomly change token to current token
                else:
                    # 10% 的概率（15% * 10% = 1.5%）保留原词
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    # 获取当前词的索引，若词不在词汇表中则使用未知标记索引

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                # 将原始词的索引添加到标签列表，若词不在词汇表中则使用未知标记索引

            else:
                # 85% 的概率保留原词
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                # 获取当前词的索引，若词不在词汇表中则使用未知标记索引
                output_label.append(0)
                # 标签为 0，表示该词未被替换

        return tokens, output_label
        # 返回替换后的词列表和对应的标签列表

    def random_sent(self, index):
        # 定义随机句子选择方法，用于 BERT 的下一句预测（NSP）任务
        t1, t2,s1,s2 = self.get_corpus_line(index)
        # 获取索引对应的两个句子,及其对应的相似度

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            # 以 50% 的概率返回连续句子
            return t1, t2,s1,s2, 1
            # 返回第一个句子、第二个句子和标签 1（表示连续）
        else:
            # 以 50% 的概率返回不连续句子
            t2,s2 = self.get_random_line()
            return t1, t2,s1,s2, 0
            # 返回第一个句子、随机句子和标签 0（表示不连续）

    def get_corpus_line(self, item):
        # 定义获取指定索引的语料行方法
        
        if self.on_memory:
            # 如果数据已加载到内存
            qgene,sgene,pident = self.lines[item]
            sgene = sgene.split(" ")
            pident = pident.split(" ")
            pident = [float(x)/100 for x in pident]
            len_sgene = len(sgene)

            # 返回索引对应的第一个和第二个句子,及其对应的相似度
            return sgene[:len_sgene//2],sgene[len_sgene//2:],pident[:len_sgene//2],pident[len_sgene//2:]
        else:
            # 如果数据未加载到内存
            line = self.file.__next__()
            # 读取文件下一行
            if line is None:
                # 如果文件已读到末尾
                self.file.close()
                # 关闭当前文件
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                # 重新打开文件
                line = self.file.__next__()
                # 读取第一行

            qgene,sgene,pident = line[:-1]
            sgene = sgene.split(" ")
            pident = pident.split(" ")
            pident = [float(x)/100 for x in pident]
            len_sgene = len(sgene)

            return sgene[:len_sgene//2],sgene[len_sgene//2:],pident[:len_sgene//2],pident[len_sgene//2:]
            # 返回两个句子

    def get_random_line(self):
        # 定义获取随机句子方法
        if self.on_memory:
            # 如果数据已加载到内存
            qgene,sgene,pident = self.lines[random.randrange(len(self.lines))]
            sgene = sgene.split(" ")
            pident = pident.split(" ")
            pident = [float(x)/100 for x in pident]
            len_sgene = len(sgene)
            return sgene[len_sgene//2:],pident[len_sgene//2:]
            # 从内存中随机选择一行的第二个句子

        line = self.file.__next__()
        # 读取文件下一行
        if line is None:
            # 如果文件已读到末尾
            self.file.close()
            # 关闭当前文件
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            # 重新打开文件
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                # 随机跳过若干行，限制最大跳跃行数
                self.random_file.__next__()
                # 前进随机文件指针
            line = self.random_file.__next__()
            # 读取随机文件下一行
        qgene,sgene,pident = line[:-1]
        sgene = sgene.split(" ")
        pident = pident.split(" ")
        pident = [float(x)/100 for x in pident]
        len_sgene = len(sgene)
        return sgene[len_sgene//2:],pident[len_sgene//2:]
        # 返回去掉换行符后按制表符分割的第二个句子