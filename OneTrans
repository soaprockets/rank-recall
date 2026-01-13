import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class SequentialTokenizer(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.proj = Dense(d_model)

    def call(self, x):  # x: [B, seq_len, in_dim]
        # 对每个位置共享同一个投影
        return self.proj(x)  # [B, seq_len, d_model]


class AutoSplitTokenizer(Layer):
    def __init__(self, num_T, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.d_model = d_model
        self.proj = Dense(num_T * d_model)

    def call(self, x):  # x: [B, in_dim]
        x = self.proj(x)
        return tf.reshape(x, [-1, self.num_T, self.d_model])  # [B, num_T, d_model]


class GroupWiseTokenizer(Layer):
    def __init__(self, num_T, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.proj = [Dense(d_model) for _ in range(num_T)]

    def call(self, x):  # x: [B, in_dim],  in_dim % num_T == 0
        parts = tf.split(x, self.num_T, axis=-1)                 # list of [B, in_dim/num_T]
        tokens = [p(parts[i]) for i, p in enumerate(self.proj)]  # each -> [B, d_model]
        return tf.stack(tokens, axis=1)    
# ---------------- RMSNorm ----------------
class RMSLayerNorm(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=(input_shape[-1],), initializer="ones")

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


# ---------------- Mixed FFN (tail LNS token-specific) ----------------
class MixedFFN(Layer):
    # 尾部 min(LNS, T) token 用 token-specific，其余 shared
    def __init__(self, d_model, d_ff, LNS, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.LNS = LNS
        self.W1S, self.W2S = Dense(d_ff), Dense(d_model)

        init = tf.keras.initializers.GlorotUniform()
        self.W1NS = self.add_weight("W1NS", (LNS, d_model, d_ff), initializer=init)
        self.b1NS = self.add_weight("b1NS", (LNS, d_ff), initializer="zeros")
        self.W2NS = self.add_weight("W2NS", (LNS, d_ff, d_model), initializer=init)
        self.b2NS = self.add_weight("b2NS", (LNS, d_model), initializer="zeros")

        self.act = tf.keras.activations.get(activation)

    def call(self, x):
        T = tf.shape(x)[1]
        t = tf.minimum(T, self.LNS)   # tail token-specific count
        s = T - t                     # shared count

        yS = self.W2S(self.act(self.W1S(x[:, :s])))  # [B,s,D]

        xT = x[:, s:]                                # [B,t,D]
        W1, b1 = self.W1NS[-t:], self.b1NS[-t:]
        W2, b2 = self.W2NS[-t:], self.b2NS[-t:]
        h  = self.act(tf.einsum("btd,tde->bte", xT, W1) + b1[None])
        yT = tf.einsum("btd,tde->bte", h, W2) + b2[None]  # [B,t,D]

        return tf.concat([yS, yT], axis=1)


# ---------------- Pyramid Mixed Causal Attention (Eq.14 strict) ----------------
class PyramidMixedCausalAttention(Layer):
    """
    Eq.(14) strict:
      - Q from tail set (length Lq)
      - K/V from full sequence (length L)
      - output keeps only tail (length Lq)
    Mixed parameterization:
      - tail min(L, LNS) tokens use token-specific
      - earlier tokens use shared
    """
    def __init__(self, d_model, num_heads, LNS, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.D, self.H, self.dh = d_model, num_heads, d_model // num_heads
        self.LNS = LNS

        self.WqS = Dense(d_model, use_bias=False)
        self.WkS = Dense(d_model, use_bias=False)
        self.WvS = Dense(d_model, use_bias=False)

        init = tf.keras.initializers.GlorotUniform()
        self.WqNS = self.add_weight("WqNS", (LNS, d_model, d_model), initializer=init)
        self.WkNS = self.add_weight("WkNS", (LNS, d_model, d_model), initializer=init)
        self.WvNS = self.add_weight("WvNS", (LNS, d_model, d_model), initializer=init)

        self.Wo = Dense(d_model, use_bias=False)

    def _mh(self, x):  # [B,T,D] -> [B,H,T,dh]
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        return tf.transpose(tf.reshape(x, [b, t, self.H, self.dh]), [0, 2, 1, 3])

    def _unmh(self, x):  # [B,H,T,dh] -> [B,T,D]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.D])

    def call(self, x, Lq):
        L = tf.shape(x)[1]
        t = tf.minimum(L, self.LNS)   # tail token-specific
        s = L - t                     # head shared

        xS, xT = x[:, :s], x[:, s:]

        Q = tf.concat([self.WqS(xS), tf.einsum("btd,tde->bte", xT, self.WqNS[-t:])], 1)
        K = tf.concat([self.WkS(xS), tf.einsum("btd,tde->bte", xT, self.WkNS[-t:])], 1)
        V = tf.concat([self.WvS(xS), tf.einsum("btd,tde->bte", xT, self.WvNS[-t:])], 1)

        Q = Q[:, -Lq:]  # only tail queries (all tokens in Q will be updated)

        Qh, Kh, Vh = self._mh(Q), self._mh(K), self._mh(V)
        logits = tf.matmul(Qh, Kh, transpose_b=True) * (tf.cast(self.dh, tf.float32) ** -0.5)

        # causal mask for tail queries: absolute indices [L-Lq .. L-1]
        q = tf.range(L - Lq, L)[:, None]
        k = tf.range(L)[None, :]
        logits += tf.cast(k > q, tf.float32)[None, None] * (-1e9)

        out = tf.matmul(tf.nn.softmax(logits, -1), Vh)  # [B,H,Lq,dh]
        return self.Wo(self._unmh(out))                 # [B,Lq,D]


# ---------------- OneTrans Block (auto Lq=L-1) ----------------
class OneTransBlock(Layer):
    def __init__(self, d_model, num_heads, d_ff, LNS, ln_eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.ln1, self.ln2 = RMSLayerNorm(ln_eps), RMSLayerNorm(ln_eps)
        self.mha = PyramidMixedCausalAttention(d_model, num_heads, LNS)
        self.ffn = MixedFFN(d_model, d_ff, LNS)

    def call(self, x):
        Lq = tf.shape(x)[1] - 1                  # 每层只压缩1个最旧token
        z = self.mha(self.ln1(x), Lq) + x[:, -Lq:]   # residual对齐尾部
        return self.ffn(self.ln2(z)) + z


# ---------------- Stack: compress S for LS layers ----------------
class OneTransStackPyramid(Layer):
    def __init__(self, LS, d_model, num_heads, d_ff, LNS, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [OneTransBlock(d_model, num_heads, d_ff, LNS) for _ in range(LS)]

    def call(self, x):
        h = x
        for blk in self.blocks:
            h = blk(h)
        return h  # [B,LNS,D]


# ---------------- Test ----------------
def test_onetrans():
    tf.random.set_seed(0)

    B = 2
    LS, LNS = 4, 2
    D_MODEL = 32
    NUM_HEAD = 4
    D_FF = 64

    x = tf.random.normal([B, LS + LNS, D_MODEL])

    model = OneTransStackPyramid(LS=LS, d_model=D_MODEL, num_heads=NUM_HEAD, d_ff=D_FF, LNS=LNS)

    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_mean(y ** 2)

    grads = tape.gradient(loss, model.trainable_variables)
    none = [(v.name, v.shape) for v, g in zip(model.trainable_variables, grads) if g is None]

    print("input :", x.shape)
    print("output:", y.shape)         # expect [B, LNS, D]
    print("loss  :", float(loss))
    print("none grads:", len(none), "/", len(grads))
    if none:
        print("example none:", none[:5])

    assert y.shape == (B, LNS, D_MODEL)
    assert any(g is not None for g in grads), "All gradients are None!"

if __name__ == "__main__":
    test_onetrans()
