# Tang-Poem-RNN

**Character-Level Chinese Classical Poetry Generation with SimpleRNN**

---

## Overview

This project trains a character-level recurrent neural network on a corpus of 43,000+ Tang dynasty poems to learn the statistical patterns of classical Chinese verse. Given a start token, the trained model generates new poem-like text one character at a time by repeatedly predicting the most probable next character.

The repository includes a modular training pipeline (`dataloader`, `model`, `trainer`, `main`) as well as an exploratory Jupyter notebook that walks through the original exercise implementation step by step.

---

## How It Works

Classical Chinese poetry generation is framed as a **next-character prediction** task:

```
Input sequence:   S  春  风  又  绿  江  南  岸
Target sequence:  春  风  又  绿  江  南  岸  E
```

The model learns: given the characters seen so far, what is the most likely next character? At inference time, starting from the `S` (start) token, it samples greedily until the `E` (end) token appears or the length limit is reached.

---

## Architecture

```
Input token IDs  [batch, seq_len]
        │
   Embedding layer  (vocab_size → 64)
        │
   SimpleRNN  (hidden_size = 128, return_sequences=True)
        │
   Dense  (128 → vocab_size)
        │
Output logits  [batch, seq_len, vocab_size]
```

**Loss:** sparse softmax cross-entropy, masked to ignore PAD positions  
**Optimizer:** Adam (lr = 0.001)  
**Training:** 20 epochs, batch size 16

---

## Project Structure

```
Tang-Poem-RNN/
├── main.py                             # Entry point: data → train → generate
├── dataloader.py                       # Corpus loading, vocab building, tf.data pipeline
├── model.py                            # PoemRNNModel (Embedding + SimpleRNN + Dense)
├── trainer.py                          # PoemTrainer (train loop, gradient update)
├── poems.txt                           # Training corpus: 43,000+ Tang poems
├── tangshi.txt                         # Supplementary Tang poetry data
└── poem_generation_with_RNN-exercise.ipynb   # Original exercise notebook
```

---

## Data Format

`poems.txt` stores one poem per line in the format:

```
题目:诗句内容全文
```

The dataloader strips the title, wraps the content with `S` / `E` tokens, filters out poems longer than 100 characters, discards characters appearing fewer than 5 times, and builds a shared `word2id` / `id2word` vocabulary.

---

## Sample Output

After 10 epochs on the notebook implementation (loss ≈ 5.04):

```
一年一里，花落花声。山风雨，风吹落月。山风雨，风吹落月。...
```

The model captures classical poetic vocabulary and rhythm but begins to loop — a known limitation of SimpleRNN's short-term memory. Replacing the cell with LSTM or GRU is the primary recommended improvement.

---

## Limitations and Improvements

The current SimpleRNN cell has no long-term memory, which causes the model to fall into repetitive loops after generating a few coherent phrases. Suggested improvements:

- Replace `SimpleRNNCell` with `LSTMCell` or `GRUCell`
- Add temperature sampling instead of greedy `argmax` to increase diversity
- Use a larger corpus or pre-trained character embeddings
- Train for more epochs with learning rate decay

---

## License

This repository is released for academic and non-commercial use only.

Copyright © 2024 Merlin. All rights reserved.

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
