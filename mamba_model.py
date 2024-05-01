import torch.nn as nn
import torch
import numpy as np
from mamba_ssm import Mamba

# pip install causal-conv1d>=1.2.0 mamba-ssm
# This only works with GPU for some reason, no CPU

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)  # (Batch, Embed, H, W)
        x = x.flatten(2).transpose(1, 2)  # (Batch, num_patches, Embed)
        x = self.norm(x)
        x = self.silu(x)
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self): # assumes square grid max_freq x max_freq
        super().__init__()
        
    def forward(self, x, max_freq):
        scale = 2 * np.pi / max_freq  # Scale factor for the sinusoidal function
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1) * scale

        row_indices = (torch.arange(seq_len, dtype=torch.long, device=x.device) // max_freq).unsqueeze(-1) * scale
        combined_pos = torch.cat([pos.sin(), pos.cos(), row_indices.sin(), row_indices.cos()], dim=-1)
    
        x = torch.cat([x, combined_pos.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=-1)
        return x

class ForwardSequenceTransform(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, old_patch_size, new_patch_size):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_embed_dim, out_embed_dim, kernel_size=(old_patch_size // new_patch_size) ** 2, stride=(old_patch_size // new_patch_size) ** 2)
        self.layernorm = nn.LayerNorm(out_embed_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        # Reshape and transpose the input to: (batch_size, in_embed_dim, in_seq_length)
        x = x.transpose(1, 2)

        # Apply the convolution layer
        x = self.conv(x)

        # Transpose the output to: (batch_size, out_seq_length, out_embed_dim)
        x = x.transpose(1, 2)
        x = self.silu(self.layernorm(x))
        return x

class BackwardSequenceTransform(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, old_patch_size, new_patch_size):
        super().__init__()
        self.conv = nn.Conv1d(in_embed_dim, out_embed_dim, kernel_size=(new_patch_size // old_patch_size) ** 2, stride=(new_patch_size // old_patch_size) ** 2)
        self.layernorm = nn.LayerNorm(out_embed_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        # Reshape and transpose the input to: (batch_size, in_embed_dim, in_seq_length)
        x = x.transpose(1, 2)

        # Apply the convolution layer
        x = self.conv(x)

        # Transpose the output to: (batch_size, out_seq_length, out_embed_dim)
        x = x.transpose(1, 2)
        x = self.silu(self.layernorm(x))
        return x

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, pooling_factor=1, skip=True):
        super().__init__()
        self.forward_mamba = Mamba(d_model, d_state, d_conv, expand)
        self.backward_mamba = Mamba(d_model, d_state, d_conv, expand)
        self.layer_norm = nn.LayerNorm(d_model)
        self.silu = nn.SiLU()
        self.pooling_factor = pooling_factor
        self.skip = skip

    def forward(self, x):
        forward_out = self.forward_mamba(x)
        backward_out = self.backward_mamba(x.flip(1)).flip(1)
        out = (forward_out + backward_out) / 2

        if self.pooling_factor > 1: # Mean pooling along sequence dim
            out = out.view(out.shape[0], out.shape[1] // self.pooling_factor, self.pooling_factor, -1)
            out = out.mean(dim=2)
        
        out = self.layer_norm(out)
        out = self.silu(out)

        if self.skip: # skip connection
            if self.pooling_factor > 1:
                out += x.view(x.shape[0], x.shape[1] // self.pooling_factor, self.pooling_factor, -1).mean(dim=2)
            else:
                out += x
        return out
    

class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = x.mean(dim=1)  # global average pooling
        return self.linear(x) # outputting logits, since logsoftmax/BCE_with_Logits is typically better, use bottom otherwise
        #return nn.functional.softmax(x, dim=-1) # Or sigmoid

class VisionMamba(nn.Module):
    def __init__(self, 
                 patch_size=8, # size of tiles for image
                 patch_embed_dim=192, # embedding dimension of patches
                 max_freq=32, # how often rotary embeddings complete a rotation
                 d_state=16, # State space size/expansion
                 d_conv=4, # mambda conv window size
                 expand=2, # block expansion factor
                 num_classes=2, # number of classes
                 num_layers=1, # number of bidirectional mambas
                 pooling_factor=1, # window size for pooling over the sequence
                 pooling_list=[], # which mamba block layers have pooling
                 device='cuda'):
        super().__init__()
        self.tokenizer = PatchEmbed(patch_size=patch_size, embed_dim=patch_embed_dim).to(device)
        self.position = RotaryEmbedding().to(device) # The 4s below come from size of this
        self.max_freq = max_freq
        self.mambas = nn.Sequential(
            *[BidirectionalMambaBlock(patch_embed_dim+4, d_state, d_conv, expand, pooling_factor) if i in pooling_list
              else BidirectionalMambaBlock(patch_embed_dim+4, d_state, d_conv, expand) # no pooling if not in list
              for i in range(num_layers)]
        ).to(device)
        self.classifier = Classifier(patch_embed_dim+4, num_classes).to(device)
    
    def forward(self, x): # Accepts batch of image tensors in standard pytorch format
        x = self.tokenizer(x)
        x = self.position(x, self.max_freq)
        x = self.mambas(x)
        return self.classifier(x)
    

class SwinMamba(nn.Module): # Patch sizes should be in descending order and divisble with image size and each other. Embed dims should be divisible by patch_size ** 2
    def __init__(self, patch_sizes, patch_embed_dims, img_size, d_state, d_convs, expand, num_classes, device='cuda'):
        super().__init__()
        self.img_size = img_size
        self.forward_embeddings = nn.ModuleList([PatchEmbed(3, patch_size, patch_dim).to(device) for patch_size, patch_dim in zip(patch_sizes, patch_embed_dims)])
        self.forward_transforms = nn.ModuleList([ForwardSequenceTransform(patch_embed_dims[i-1], patch_embed_dims[i], patch_sizes[i-1], patch_sizes[i]) for i in range(1, len(patch_sizes))])
        self.backward_transforms = nn.ModuleList([BackwardSequenceTransform(patch_embed_dims[i], patch_embed_dims[i-1], patch_sizes[i], patch_sizes[i-1]) for i in range(1, len(patch_sizes))])
        self.position = RotaryEmbedding().to(device) # The 4 below come from size of this
        self.forward_mambas = nn.ModuleList([BidirectionalMambaBlock(patch_dim+4, d_state, d_conv, expand, skip=False).to(device) for d_conv, patch_dim in zip(d_convs, patch_embed_dims)])
        self.backward_mambas = nn.ModuleList([BidirectionalMambaBlock(patch_dim+4, d_state, d_conv, expand, skip=False).to(device) for d_conv, patch_dim in zip(d_convs[:-1], patch_embed_dims[:-1])])
        self.classifier = Classifier(patch_embed_dims[0]+4, num_classes).to(device)
    
    def forward(self, x):
        intermediates = []
        for i in range(len(self.forward_embeddings)):
            embed = self.forward_embeddings[i]
            if i == 0:
                embed = self.position(embed(x), self.img_size // embed.patch_size)
                intermediates.append(self.forward_mambas[i](embed))
            else:
                new_seq = intermediates[-1][:, :, :-4]
                new_seq = self.forward_transforms[i-1](new_seq)
                new_seq = self.position(embed(x) + new_seq, self.img_size // embed.patch_size)
                intermediates.append(self.forward_mambas[i](new_seq))
        current = intermediates[-1][:, :, :-4]
        current = self.backward_transforms[-1](current)
        current = self.position(current, self.img_size // self.forward_embeddings[-2].patch_size)
        for i in reversed(range(len(self.backward_mambas))):
            current = self.backward_mambas[i](current + intermediates[i])
            if i > 0:
                current = intermediates[i][:, :, :-4]
                current = self.backward_transforms[i-1](current)
                current = self.position(current, self.img_size // self.forward_embeddings[i-1].patch_size)
        
        return self.classifier(current)