# A-simple-GPT

A decoder-only GPTModel following the paper "Attention is All You Need"
`https://arxiv.org/abs/1706.03762`

### Dataset
```https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt```
~1M tokens ( character level )

### Model Parameters
~10M trainable parameters

### Hyperparameters used
batch_size = 64< br / >
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


