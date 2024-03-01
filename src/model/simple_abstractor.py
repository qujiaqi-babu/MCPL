import torch
import torch.nn as nn

class Cross_Attention(nn.Module):
    def __init__(self,
                 dim=768,   # 输入token的dim
                 num_heads=12,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x_q, x_k, x_v):
        # [batch_size, num_patches + 1, total_embed_dim]
        # qkv(): -> [batch_size, num_patches + 1, total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # permute: -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        B, N, C = x_q.shape
        q = self.qkv_q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B, N, C = x_k.shape
        k = self.qkv_k(x_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B, N, C = x_v.shape
        v = self.qkv_v(x_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(x_q.shape[0], x_q.shape[1], x_q.shape[2])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Simple_Abstractor(nn.Module):
    def __init__(self,
                 dim=768,   # 输入token的dim
                 num_heads=12,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 learnable_token_num=36,
                 clip_feature_num=49,
                 input_dim=2048,
                 output_dim=768):
        super(Simple_Abstractor, self).__init__()
        self.in_fc = nn.Linear(input_dim, dim)
        self.learnable_token = nn.Parameter(torch.zeros(1, learnable_token_num, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_feature_num+learnable_token_num, dim))  # 拼接
        # self.pos_embed = nn.Parameter(torch.zeros(1, clip_feature_num, dim))  # 不拼接
        self.cross_attention = Cross_Attention(
            dim,   # 输入token的dim
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop_ratio,
            proj_drop_ratio)
        self.out_fc = nn.Linear(dim, output_dim)
        # self.norm = nn.LayerNorm(output_dim)
        nn.init.trunc_normal_(self.learnable_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x_img_feat):
        B, N, C = x_img_feat.shape  # [B, 49, 2048]
        x_img_feat = self.in_fc(x_img_feat)  # [B, 49, 768]
        x_q = self.learnable_token.expand(B, -1, -1)  # [B, 36, 768]
        x_k = x_v = torch.cat((x_img_feat, x_q), dim=1) + self.pos_embed  # [B, 85, 768] 拼接
        # x_k = x_v = x_img_feat + self.pos_embed  # [B, 49, 768] 不拼接
        output = self.cross_attention(x_q, x_k, x_v)  # [B, 36, 768]
        output = self.out_fc(output)
        # output = self.norm(output)
        return output


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)