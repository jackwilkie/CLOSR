'''
Contrastive Loss for Outlier Detection

Created on: 11/04/24
'''

import torch as T
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from custom_transformers.encoders import Mlp
from self_supervised.tabular.embedding import FeatureEmbedding
from data_read.images.cutmix import make_batch_mix
from metric_learning.contrastive_anomaly import MLPBlock
from metric_learning.losses.triplet_losses import TripletLossWrapper
from mlps.feed_forward import ContrastiveMLP, make_mlp
from self_supervised.tabular import augmentations

from timm.models.layers import DropPath
from typing import Optional
from einops import rearrange
from custom_transformers.encoders import Encoder

from model_training.losses import BaseLoss

def cossim(a, b = None):
    a_norm = F.normalize(a, p=2, dim=-1)
    
    if b is not None:
        b_norm = F.normalize(b, p=2, dim=-1)
    else:
        b_norm = a_norm.clone()
        
    # Compute cosine similarity
    # Using torch.mm for matrix multiplication because A_norm and B_norm.T are normalized
    similarity = T.mm(a_norm, b_norm.T)

    # Since cosine distance = 1 - cosine similarity
    cosine_distance = (1 - similarity)/2
    
    return cosine_distance


class COWrapper(BaseLoss):
    def forward(self, model, x, y, mixed_precision, training):
        logits, y_pred, y_true = self.get_model_input(model, x, y, mixed_precision)
        return self.loss(logits, model), self.calc_metric(logits.clone().detach(), y_true)


class _ContrastiveOutlier(nn.Module):
    def __init__(
        self, 
        m: float = 1.,
        squared: bool = True,
        alpha_similar: float = 1.,
        alpha_dissimilar: float = 1.,
        alpha_reconstructive: float =  1.,
        embedded_reconstruction: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        
        # -- init parameters        
        self.m = m 
        self.squared = squared
        self.eps = eps 
        self.embedded_reconstruction = embedded_reconstruction
        self.fraction_positive_pairs = 0.
        
        # -- get contribution of each loss
        alpha_scalar = 1/(alpha_reconstructive + alpha_similar + alpha_dissimilar)
        self.recon_scale = alpha_reconstructive * alpha_scalar
        self.sim_scale = alpha_similar * alpha_scalar
        self.dissim_scale = alpha_similar * alpha_scalar
        

    def forward( 
        self, 
        x: Tensor,
        model,
    ) -> Tensor:
        
        # -- get clean and noisy features
        z, z_pred, z_cont, z_bar_cont = model.forward_pretrain(x)
        
        # -- contrastive loss
        sim_dists = cossim(z_cont, z_cont)
        if self.squared:
            sim_dists = T.pow(sim_dists, 2)
        
        n_sim = T.sum(T.greater(sim_dists, self.eps).float()) # number of similar pairs
        sim_loss = T.sum(sim_dists)/(n_sim+self.eps) # mean loss for similar pairs        

        # get dists fo dissimilar pairs
        
        dissim_dists = cossim(z_cont, z_bar_cont)
        if self.squared:
            dissim_dists = T.pow(dissim_dists, 2)
            
        dissim_loss = F.relu(self.m - dissim_dists)
        n_dissim = T.sum(T.greater(dissim_loss,self.eps).float())
        dissim_loss = T.sum(dissim_loss)/(n_dissim+self.eps)
        
        #print(f'dissim_loss: {dissim_loss}')
        self.fraction_positive_pairs= n_dissim/ (z_bar_cont.size(0) * z_cont.size(0))
        
        # -- reconstructive loss
        recon_loss = T.mean(cossim(z_pred, z))
        
        return ((self.sim_scale * sim_loss) + (self.dissim_scale * sim_loss) + (self.recon_scale * recon_loss))
    
    def get_fraction_pos(self):
        return self.fraction_positive_pairs


def ContrastiveOutlier(*args, **kwargs):
    return COWrapper(_ContrastiveOutlier, *args, **kwargs)

class COModel(nn.Module):
    def __init__(
        self,
        d_model,
        n_features,
        n_cat,
        num_cats,
        d_contrastive,
        tokenised_model: bool =  False,
        noisy_head: bool = True,
        cutmix_alpha = 0.3,
        mixup_alpha = 0.3,
        p_cutmix = 0.5,
        jitter_std = 0.1,
        p_jitter = 1.0,
        use_jitter: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        self.tokenised_model = tokenised_model
            
        # define patch embeddings
        self.embedding_layer = FeatureEmbedding(
            d_model = d_model,
            n_features = n_features,
            n_cat = n_cat,
            num_cats = num_cats,
            feature_transform = nn.Identity,
            embedding_transform = nn.Identity,
        ) if tokenised_model else nn.Identity()
        
        self.blocks = nn.Identity()

        head_in_dim = d_model
        assert head_in_dim is not None, 'No head in dim'
        
        self.contrastive_head = Mlp(
            in_features= head_in_dim,
            hidden_features = int(4*head_in_dim),
            out_features = d_contrastive
        )
        
        if noisy_head:
            self.noisy_contrastive_head = Mlp(
                in_features= head_in_dim,
                hidden_features = int(4*head_in_dim),
                out_features = d_contrastive
            )
        else:
            self.noisy_contrastive_head = self.contrastive_head
            
        self.recon_head = Mlp(
            in_features= head_in_dim,
            hidden_features = int(4*head_in_dim),
            out_features = head_in_dim
        )
        
        '''
        self.aug = make_batch_mix(
            n_classes = None,
            cutmix_alpha = cutmix_alpha,
            mixup_alpha = mixup_alpha,
            p_cutmix = p_cutmix,
            use_jitter = use_jitter,
            p_jitter = p_jitter,
            jitter_mean = 0.0,
            jitter_std = jitter_std,
        )   
        '''
        self.aug = augmentations.AugmentationLayer(
            augmentations.MixMix(alpha = mixup_alpha, p_sample= p_cutmix, p_feature = 1, p_mixup = 0.5),
            augmentations.Jitter(var= jitter_std, p_sample = p_jitter, p_feature = 1.0),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.embed_input(x)
        x = self.forward_blocks(x)
        return self.contrastive_head(x)
    
    def forward_blocks(self, z: Tensor) -> Tensor:
        z = self.blocks(z)
        
        if self.tokenised_model:
            z = T.mean(z, dim = 1)
        
        return z
    
    def forward_pretrain(self, x: Tensor) -> Tensor:
        x = self.embed_input(x)
        x_bar = self.aug(x)
        
        z = self.forward_blocks(x)
        z_bar = self.forward_blocks(x_bar)
        z_cont = self.contrastive_head(z)
        z_bar_cont = self.noisy_contrastive_head(z_bar)
        
        z_pred = self.recon_head(z_bar)
        
        return z, z_pred, z_cont, z_bar_cont
        
    def embed_input(self, x: Tensor) -> Tensor:
        if self.tokenised_model:
            x = self.embedding_layer(x)
            x = rearrange(x, 'B C D -> B (C D)')
            
        return x    

    def noisy_proj(self, z: Tensor) -> Tensor:
        return self.noisy_contrastive_head(z)
        
    def contrastive_proj(self, z: Tensor) -> Tensor:
        return self.contrastive_head(z)
    
    def embed_recon_proj(self, z: Tensor) -> Tensor:
        return self.recon_head(z)

    
class RegularContrastiveOutlier(COModel):
    def __init__(
        self, 
        *args, 
        depth: int = 4,
        width: int = 1024,
        residual: bool = True,
        activation = nn.ReLU,
        dropout = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, d_model = width, tokenised_model=False)
        
        self.blocks = make_mlp(
            d_in = self.n_features,
            neurons = [width] * depth,
            activation = activation,
            dropout = dropout,
            residual=residual,
            gated = False,
            norm_layer= nn.Identity,
            dropout_layer = nn.Dropout,
            final_layer_activation = activation,
            layer_scale= None,
            block_norm = None,
        )
        
        
class TransformerOutlier(COModel):
    def __init__(
        self, 
        *args, 
        d_model: int,
        n_heads: int,
        n_blocks: int = 4,
        mlp_ratio: float = 4,
        norm_layer = nn.LayerNorm,
        act_layer = nn.GELU,
        dropout_layer = nn.Dropout,
        droppath_layer = DropPath,
        linear_layer = nn.Linear, 
        drop_rate: float = 0.0,
        attn_drop_rate: Optional[float] = None,
        droppath_rate: float = 0.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            *args, 
            d_model = d_model,
            tokenised_model= True,
            **kwargs,
        )
        
        self.blocks = self.blocks = Encoder(
            d_model = d_model, 
            n_blocks = n_blocks,
            n_heads = n_heads, 
            mlp_ratio = mlp_ratio, 
            qkv_bias = qkv_bias,
            qk_scale = qk_scale,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            norm_layer = norm_layer,
            dropout_layer = dropout_layer,
            droppath_layer = droppath_layer,
            linear_layer = linear_layer,
            act_layer = act_layer,
            drop_path_rate = droppath_rate,
        )