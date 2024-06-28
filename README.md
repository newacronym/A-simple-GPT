# A-simple-GPT

A decoder-only GPTModel following the paper "Attention is All You Need"
`https://arxiv.org/abs/1706.03762`

### Dataset
```https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt```
~1M tokens ( character level )

### Model Parameters
~10M trainable parameters

### Hyperparameters used
block_size = 256 <br/>
n_embd = 384 <br/>
n_heads = 6 <br/>
n_layers = 6 <br/>

NOTE:
It's a decoder-only model <br/>
It only has self-attention and not cross-attention
