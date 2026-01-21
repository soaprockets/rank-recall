import tensorflow as tf 
from tensorflow.keras.layers import *

class GatedSumPooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate_dense = Dense(1, activation='sigmoid')
    def call(self, x):
        g = self.gate_dense(x)  # (B, T, 1)
        x = tf.reduce_sum(x * g, axis=1) # (B, D)
        return x # (B, D)



# independent MLP for object feature
class ObjectMlp(Layer):
    def __init__(self, d_model, expansion_factor, **kwargs):
        super().__init__(**kwargs)
        d_ff = d_model * expansion_factor
        self.fc1 = Dense(d_ff)
        self.fc2 = Dense(d_model)
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

 
class RMSLayerNorm(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon
    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=(input_shape[-1],), initializer="ones")
    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale



class CausalSelfAttention(Layer):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = Dense(d_model, use_bias=False)
        self.k_proj = Dense(d_model, use_bias=False)
        self.v_proj = Dense(d_model, use_bias=False)
        self.out_proj = Dense(d_model, use_bias=False)

    def call(self, x):
        # x: [B, T, D]
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t):
            t = tf.reshape(t, [B, T, self.n_heads, self.d_head])
            return tf.transpose(t, [0, 2, 1, 3])  # [B, H, T, Dh]

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = tf.cast(self.d_head, x.dtype) ** -0.5
        att = tf.matmul(q, k, transpose_b=True) * scale  # [B, H, T, T]

        mask = tf.linalg.band_part(tf.ones([T, T], dtype=x.dtype), -1, 0)
        att = att + (1.0 - mask)[None, None, :, :] * tf.cast(-1e9, x.dtype)

        w = tf.nn.softmax(att, axis=-1)
        y = tf.matmul(w, v)                              # [B, H, T, Dh]
        y = tf.transpose(y, [0, 2, 1, 3])                # [B, T, H, Dh]
        y = tf.reshape(y, [B, T, self.d_model])          # [B, T, D]
        return self.out_proj(y)


class Decoder(Layer):
    def __init__(self,d_model,num_heads,d_ff,**kwargs):
        super().__init__(**kwargs)
        self.norm1 = RMSLayerNorm()
        self.norm2 = RMSLayerNorm()
        self.mha = CausalSelfAttention(d_model,num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff,activation='gelu'),
            Dense(d_model)
        ])
        self.dropout = Dropout(0.1)
    def call(self,x):
        x = self.dropout(self.norm1(x + self.mha(x))) # residual connection
        x = self.dropout(self.norm2(x + self.ffn(x))) 
        return x
    

        
class TaskAwareDecoder(Layer):
    def __init__(self,d_model,num_heads,d_ff,num_experts,top_k,**kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.mha = CausalSelfAttention(d_model,num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff,activation='gelu'),
            Dense(d_model)
        ])
        self.norm1,self.norm2 = RMSLayerNorm(),RMSLayerNorm()
        self.dropout = Dropout(0.1)
        self.ffn_g = tf.keras.Sequential([
            Dense(d_ff,activation='gelu'),
            Dense(num_experts),
            Activation('Softmax')
        ])
        self.ffn_expert = [tf.keras.Sequential([Dense(d_ff, activation='gelu'),
                                        Dense(d_model)])
                   for _ in range(num_experts)]

        self.mlp = Dense(d_model*(num_experts+1),activation='gelu') # ffn_gate + ffn_expert
    def call(self,x):
        x = self.dropout(self.norm1(x + self.mha(x)))

        h = self.mlp(x)
        parts = tf.split(h, self.num_experts + 1, axis=-1)
        x_gate, x_expert_in = parts[0], parts[1:]   # x_expert_in: list(E) each [B,T,D]

        # experts: list(E)[B,T,D] -> [B,T,E,D]
        experts = [self.ffn_expert[i](x_expert_in[i]) for i in range(self.num_experts)]
        experts = tf.stack(experts, axis=2)

        # gate logits: [B,T,E]（不要 split）
        g = self.ffn_g(x_gate)

        if self.top_k < self.num_experts:
            v, idx = tf.math.top_k(g, k=self.top_k)                 # v:[B,T,K], idx:[B,T,K]
            sel = tf.gather(experts, idx, batch_dims=2, axis=2)     # [B,T,K,D]
            w = tf.nn.softmax(v, axis=-1)[..., None]                # [B,T,K,1]
            mix = tf.reduce_sum(w * sel, axis=2)                    # [B,T,D]
        else:
            w = tf.nn.softmax(g, axis=-1)[..., None]                # [B,T,E,1]
            mix = tf.reduce_sum(w * experts, axis=2)                # [B,T,D]

        x = self.dropout(self.norm2(x + self.ffn(mix)))
        return x
        
        
        
class MPFormer(Layer):
    def __init__(self,d_model,num_heads,d_ff,num_experts,top_k,n_task,**kwargs):
        super().__init__(**kwargs)
        self.layers = [
            Decoder(d_model,num_heads,d_ff),
            Decoder(d_model,num_heads,d_ff),
            TaskAwareDecoder(d_model,num_heads,d_ff,num_experts,top_k),
            Decoder(d_model,num_heads,d_ff)
        ]
        self.task_mlp = [Dense(d_model) for _ in range(n_task)]
    def call(self,x):
        for layer in self.layers:
            x = layer(x)
        x = [layer(x) for layer in self.task_mlp]
        return x

if __name__ == '__main__':
    inputs = Input(shape=(32,))
    mpformer = MPFormer(d_model=32,num_heads=4,d_ff=128,num_experts=4,top_k=2,n_task=1)
    outputs = mpformer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
