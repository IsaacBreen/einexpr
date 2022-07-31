## Case study: Transformers

Let's take some *excellent* code from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) and see what makes it so special.

```py
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
att = self.attn_dropout(att)
y = att @ v
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

I've omitted comments. Without them, is it possible to understand what's going on?
