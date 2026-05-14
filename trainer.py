# trainer.py
import tensorflow as tf

class PoemTrainer:
    def __init__(self, model, learning_rate=0.0005):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def _compute_loss(self, logits, labels, seqlen):
        """计算损失函数"""
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        # 使用序列长度进行掩码，避免填充部分的损失计算
        mask = tf.sequence_mask(seqlen, tf.shape(losses)[1], dtype=tf.float32)
        losses = losses * mask
        return tf.reduce_mean(losses)

    @tf.function
    def train_step(self, x, y, seqlen):
        """单步训练"""
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self._compute_loss(logits, y, seqlen)

        # 计算梯度
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # 更新参数
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss

    def train(self, dataset, epochs):
        """完整训练过程"""
        for epoch in range(epochs):
            total_loss = 0
            steps = 0

            for x, y, seqlen in dataset:
                loss = self.train_step(x, y, seqlen)
                total_loss += loss
                steps += 1

                if steps % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {steps}, '
                          f'Loss: {loss:.4f}')

            avg_loss = total_loss / steps
            print(f'Epoch {epoch + 1} completed. '
                  f'Average loss: {avg_loss:.4f}')