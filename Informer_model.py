class InputEmbedding(nn.Module):
    def __init__(self, dim_model: int):
        super(InputEmbedding, self).__init__()
        # The input to the linear layer is expected to be 1-dimensional (a single feature)
        self.linear = nn.Linear(1, dim_model)

    def forward(self, x):
        # Reshape the input to be 2-dimensional with a single feature dimension
        x = x.unsqueeze(-1)
        return self.linear(x) * math.sqrt(self.linear.out_features)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, dim_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dropout: float):
        super(ProbSparseSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.d_k = dim_model // num_heads

        self.w_q = nn.Linear(dim_model, dim_model)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)
        self.w_o = nn.Linear(dim_model, dim_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ProbSparse attention mechanism
        U = q.size(2)  # Query sequence length
        M = int(math.ceil(math.log(U) ** 2))  # Number of sampled keys

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Sampling keys based on scores
        sampled_keys = torch.topk(scores, M, dim=-1)[1]

        sparse_k = k.gather(2, sampled_keys.expand(-1, -1, -1, self.d_k))
        sparse_v = v.gather(2, sampled_keys.expand(-1, -1, -1, self.d_k))

        scores = torch.matmul(q, sparse_k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, sparse_v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.w_o(x), p_attn

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, dim_model: int, dim_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff)
        self.linear_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, features: int, multi_head_attention: ProbSparseSelfAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = multi_head_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int, seq_len: int, distilling_layers: int):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        self.distillation_layer = DistillingLayer(features, seq_len, distilling_layers)

    def forward(self, x, mask):
        x = self.distillation_layer(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DistillingLayer(nn.Module):
    def __init__(self, features: int, seq_len: int, distilling_layers: int):
        super(DistillingLayer, self).__init__()
        self.distilling_layers = distilling_layers
        self.distillation_layers = nn.ModuleList()
        for i in range(distilling_layers):
            self.distillation_layers.append(nn.Linear(seq_len // (2 ** i), seq_len // (2 ** (i + 1))))

    def forward(self, x):
        for layer in self.distillation_layers:
            x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention: ProbSparseSelfAttention, cross_attention: ProbSparseSelfAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, memory, memory, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, memory, src_mask, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, dim_model: int, output_size: int):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(dim_model, output_size)

    def forward(self, x):
        return self.projection(x)

class Informer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super(Informer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def forward(self, src, tgt, src_mask, tgt_mask=None):
        src = self.src_pos(self.src_embed(src))
        tgt = self.tgt_pos(self.tgt_embed(tgt))
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
        return self.projection_layer(output)

def build_informer(seq_len: int, dim_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, dim_ff: int = 2048, output_size: int = 1, distilling_layers: int = 2) -> Informer:
    src_embed = InputEmbedding(dim_model)
    tgt_embed = InputEmbedding(dim_model)

    encoder_blocks = []
    decoder_blocks = []

    for i in range(N):
        multi_head_attention = ProbSparseSelfAttention(dim_model, h, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, dim_ff, dropout)
        encoder_blocks.append(EncoderBlock(dim_model, multi_head_attention, feed_forward_block, dropout))

    encoder = Encoder(nn.ModuleList(encoder_blocks), dim_model, seq_len, distilling_layers)

    for i in range(N):
        multi_head_attention = ProbSparseSelfAttention(dim_model, h, dropout)
        cross_attention_block = ProbSparseSelfAttention(dim_model, h, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, dim_ff, dropout)
        decoder_blocks.append(DecoderBlock(dim_model, multi_head_attention, cross_attention_block, feed_forward_block, dropout))

    decoder = Decoder(nn.ModuleList(decoder_blocks), dim_model)

    src_pos = PositionalEncoding(dim_model, seq_len, dropout)
    tgt_pos = PositionalEncoding(dim_model, seq_len, dropout)

    projection_layer = ProjectionLayer(dim_model, output_size)

    informer = Informer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in informer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return informer
