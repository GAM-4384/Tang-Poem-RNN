# dataloader.py
import tensorflow as tf
import collections


class PoemDataLoader:
    def __init__(self):
        self.start_token = 'S'
        self.end_token = 'E'
        self.max_length = 100  # 限制最大长度

    def process_poems(self, file_path):
        """直接处理poems.txt文件"""
        poems = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line) < self.max_length:  # 只保留较短的诗句
                    content = [self.start_token] + list(line) + [self.end_token]
                    poems.append(content)

        # 构建词表
        words = []
        for poem in poems:
            words.extend(poem)

        counter = collections.Counter(words)
        # 只保留出现次数大于5次的字
        words = [word for word, count in counter.items() if count > 5]
        words = ['PAD', 'UNK'] + words

        word2id = {word: idx for idx, word in enumerate(words)}
        id2word = {idx: word for idx, word in enumerate(words)}

        # 将诗句转换为ID序列
        poem_vectors = []
        for poem in poems:
            vector = [word2id.get(word, word2id['UNK']) for word in poem]
            poem_vectors.append(vector)

        return poem_vectors, word2id, id2word

    def create_dataset(self, file_path, batch_size=16, buffer_size=1000):
        """创建训练用的数据集"""
        poem_vectors, word2id, id2word = self.process_poems(file_path)

        # 创建序列长度列表
        seq_lengths = [len(vector) for vector in poem_vectors]

        # 创建tensorflow数据集
        ds = tf.data.Dataset.from_generator(
            lambda: [(vec, length) for vec, length in zip(poem_vectors, seq_lengths)],
            (tf.int64, tf.int64),
            (tf.TensorShape([None]), tf.TensorShape([]))
        )

        # 数据集处理流水线
        ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]))
        )
        ds = ds.map(lambda x, seqlen: (x[:, :-1], x[:, 1:], seqlen - 1))

        return ds, word2id, id2word