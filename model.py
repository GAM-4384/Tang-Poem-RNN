# model.py
import tensorflow as tf
from tensorflow import keras


class PoemRNNModel(keras.Model):
    def __init__(self, vocab_size, embedding_dim=64, rnn_units=128):
        super(PoemRNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units

        # 定义各层
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim,
            batch_input_shape=[None, None]
        )
        self.rnn_cell = tf.keras.layers.SimpleRNNCell(rnn_units)
        self.rnn = tf.keras.layers.RNN(
            self.rnn_cell,
            return_sequences=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    @tf.function
    def call(self, inputs):
        # 词嵌入
        x = self.embedding(inputs)
        # RNN处理
        x = self.rnn(x)
        # 输出层
        logits = self.dense(x)
        return logits

    @tf.function
    def get_next_token(self, x, state):
        """生成下一个字的预测"""
        # 词嵌入
        x = self.embedding(x)
        # RNN单步处理
        h, state = self.rnn_cell(x, state)
        # 输出层
        logits = self.dense(h)
        # 选择最可能的字
        predicted_id = tf.argmax(logits, axis=-1)
        return predicted_id, state

    def generate_text(self, start_token_id, max_length, temperature=1.0):
        """生成诗句"""
        # 初始化RNN状态
        state = tf.random.normal([1, self.rnn_units])
        tokens = [start_token_id]

        for _ in range(max_length):
            current_token = tf.constant([tokens[-1]], dtype=tf.int32)
            next_token, state = self.get_next_token(current_token, state)
            tokens.append(int(next_token))

        return tokens