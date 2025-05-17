import torch
from torch.utils.data import DataLoader
from blast.dataset import BERTDataset, WordVocab

def main():
    # 1. 加载词汇表
    print("Loading Vocab...")
    vocab = WordVocab.load_vocab("vocab.pkl")
    print(f"Vocab Size: {len(vocab)}")

    # 2. 创建数据集
    print("Creating Dataset...")
    dataset = BERTDataset(
        corpus_path="output.txt",
        vocab=vocab,
        seq_len=512,  # 使用默认的序列长度
        encoding="utf-8",
        on_memory=True  # 将数据加载到内存中
    )
    print(f"Dataset Size: {len(dataset)}")

    # 3. 创建数据加载器
    print("Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 使用较小的batch_size便于测试
        num_workers=0,  # 使用单进程加载数据
        shuffle=True
    )

    # 4. 遍历数据加载器
    print("\nIterating through DataLoader:")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i + 1}:")
        print(f"bert_input shape: {batch['bert_input'].shape}")
        print(f"bert_label shape: {batch['bert_label'].shape}")
        print(f"segment_label shape: {batch['segment_label'].shape}")
        print(f"is_next shape: {batch['is_next'].shape}")
        
        # 打印第一个样本的详细信息
        if i == 0:
            print("\nFirst sample details:")
            print(f"bert_input: {batch['bert_input'][0]}")
            print(f"bert_label: {batch['bert_label'][0]}")
            print(f"segment_label: {batch['segment_label'][0]}")
            print(f"is_next: {batch['is_next'][0]}")
            print(f"s_label: {batch['s_label'][0]}")

        # 只打印前3个batch作为示例
        if i >= 2:
            break

if __name__ == "__main__":
    main() 