# main.py
from dataloader import PoemDataLoader
from model import PoemRNNModel
from trainer import PoemTrainer


def main():
    # 初始化数据加载器
    dataloader = PoemDataLoader()

    # 创建数据集
    dataset, word2id, id2word = dataloader.create_dataset(
        'E:\\My struggle\\poem\\poems.txt',
        batch_size=16  # 显著减小batch size
    )

    # 创建模型
    model = PoemRNNModel(
        vocab_size=len(word2id),
        embedding_dim=64,  # 减小embedding维度
        rnn_units=128  # 减小RNN单元数量
    )

    # 创建训练器
    trainer = PoemTrainer(model, learning_rate=0.001)

    # 开始训练
    trainer.train(dataset, epochs=20)

    # 生成诗句示例
    start_token_id = word2id['S']
    generated_ids = model.generate_text(
        start_token_id,
        max_length=50
    )

    # 将ID转换回文字并移除特殊标记
    generated_poem = ''.join([id2word[id] for id in generated_ids])
    generated_poem = generated_poem.replace('S', '').replace('E', '')
    print("生成的诗句:", generated_poem)


if __name__ == "__main__":
    main()