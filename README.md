# A-simple-GPT

A decoder-only GPTModel following the paper "Attention is All You Need"
`https://arxiv.org/abs/1706.03762`

### Dataset
```https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt```
~1M tokens ( character level )

### Model Parameters
~10M trainable parameters

### Hyperparameters used
vocab_size = 65 <br/>
block_size = 256 <br/>
n_embd = 384 <br/>
n_heads = 6 <br/>
n_layers = 6 <br/>

NOTE:
It's a decoder-only model <br/>
It only has self-attention and not cross-attention

![image](https://github.com/newacronym/A-simple-GPT/assets/51745787/45f0d35b-8f21-4f5b-8d32-d5fdcb68bf7e)
