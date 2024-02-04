import torch
import torch.nn as nn
from DiffTransformer import TransformerEncoder


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_skip_connection=None,
        use_new_ffn=None,
        ref_in_dim=None,
        ref_out_dim=None,
        use_query_emb=None,
        num_query_emb=None,
        cfg=None,
    ):
        super().__init__()

        self.encoder_layer = (
            encoder_layer if encoder_layer is not None else cfg.encoder_layer
        )
        self.encoder_hidden = (
            encoder_hidden if encoder_hidden is not None else cfg.encoder_hidden
        )
        self.encoder_head = (
            encoder_head if encoder_head is not None else cfg.encoder_head
        )
        self.conv_filter_size = (
            conv_filter_size if conv_filter_size is not None else cfg.conv_filter_size
        )
        self.conv_kernel_size = (
            conv_kernel_size if conv_kernel_size is not None else cfg.conv_kernel_size
        )
        self.encoder_dropout = (
            encoder_dropout if encoder_dropout is not None else cfg.encoder_dropout
        )
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn
        self.in_dim = ref_in_dim if ref_in_dim is not None else cfg.ref_in_dim
        self.out_dim = ref_out_dim if ref_out_dim is not None else cfg.ref_out_dim
        self.use_query_emb = (
            use_query_emb if use_query_emb is not None else cfg.use_query_emb
        )
        self.num_query_emb = (
            num_query_emb if num_query_emb is not None else cfg.num_query_emb
        )

        if self.in_dim != self.encoder_hidden:
            self.in_linear = nn.Linear(self.in_dim, self.encoder_hidden)
            self.in_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.in_dim = None

        if self.out_dim != self.encoder_hidden:
            self.out_linear = nn.Linear(self.encoder_hidden, self.out_dim)
            self.out_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.out_linear = None

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            encoder_hidden=self.encoder_hidden,
            encoder_head=self.encoder_head,
            conv_kernel_size=self.conv_kernel_size,
            conv_filter_size=self.conv_filter_size,
            encoder_dropout=self.encoder_dropout,
            use_new_ffn=self.use_new_ffn,
            use_cln=False,
            use_skip_connection=False,
            add_diff_step=False,
        )

        if self.use_query_emb:
            self.query_embs = nn.Embedding(self.num_query_emb, self.encoder_hidden)
            self.query_attn = nn.MultiheadAttention(
                self.encoder_hidden, self.encoder_hidden // 64, batch_first=True
            )

    def forward(self, x_ref, key_padding_mask=None):
        # x_ref: (B, T, d_ref)
        # key_padding_mask: (B, T)
        # return speaker embedding: x_spk
        # if self.use_query_embs: shape is (B, N_query, d_out)
        # else: shape is (B, 1, d_out)

        if self.in_linear != None:
            x = self.in_linear(x_ref)

        x = self.transformer_encoder(
            x, key_padding_mask=key_padding_mask, condition=None, diffusion_step=None
        )

        if self.use_query_emb:
            spk_query_emb = self.query_embs(
                torch.arange(self.num_query_emb).to(x.device)
            ).repeat(x.shape[0], 1, 1)
            spk_embs, _ = self.query_attn(
                query=spk_query_emb,
                key=x,
                value=x,
                key_padding_mask=(
                    ~(key_padding_mask.bool()) if key_padding_mask is not None else None
                ),
            )

            if self.out_linear != None:
                spk_embs = self.out_linear(spk_embs)

        else:
            spk_query_emb = None

        return spk_embs, x
