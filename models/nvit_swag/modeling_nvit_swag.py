# --------------------------------------------------------
# References:
# timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
# SWAG: https://github.com/facebookresearch/SWAG/blob/main/models/vision_transformer.py
# --------------------------------------------------------
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Mapping, NamedTuple, Optional, Union

import torch
import torch.nn as nn


NORMALIZE_L2 = "l2"


LayerNorm = partial(nn.LayerNorm, eps=1e-6)


def is_pos_int(number: int) -> bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


class ConvStemLayer(NamedTuple):
    kernel: int
    stride: int
    out_channels: int


def lecun_normal_init(tensor, fan_in):
    nn.init.trunc_normal_(tensor, std=math.sqrt(1 / fan_in))


def get_same_padding_for_kernel_size(kernel_size):
    """
    Returns the required padding for "same" style convolutions
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"Only odd sized kernels are supported, got {kernel_size}")
    return (kernel_size - 1) // 2


class VisionTransformerHead(nn.Module):
    def __init__(
        self,
        in_plane: int,
        num_classes: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        normalize_inputs: Optional[str] = None,
    ):
        """
        Args:
            in_plane: Input size for the fully connected layer
            num_classes: Number of output classes for the head
            hidden_dim: If not None, a hidden layer with the specific dimension is added
            normalize_inputs: If specified, normalize the inputs using the specified
                method. Supports "l2" normalization.
        """
        super().__init__()

        if normalize_inputs is not None and normalize_inputs != NORMALIZE_L2:
            raise ValueError(
                f"Unsupported value for normalize_inputs: {normalize_inputs}"
            )

        if num_classes is None:
            layers = []
        elif hidden_dim is None:
            layers = [("head", nn.Linear(in_plane, num_classes))]
        else:
            layers = [
                ("pre_logits", nn.Linear(in_plane, hidden_dim)),
                ("act", nn.Tanh()),
                ("head", nn.Linear(hidden_dim, num_classes)),
            ]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.normalize_inputs = normalize_inputs
        self.init_weights()

    def init_weights(self):
        if hasattr(self.layers, "pre_logits"):
            lecun_normal_init(
                self.layers.pre_logits.weight, fan_in=self.layers.pre_logits.in_features
            )
            nn.init.zeros_(self.layers.pre_logits.bias)
        if hasattr(self.layers, "head"):
            nn.init.zeros_(self.layers.head.weight)
            nn.init.zeros_(self.layers.head.bias)

    def forward(self, x):
        if self.normalize_inputs is not None:
            if self.normalize_inputs == NORMALIZE_L2:
                x = nn.functional.normalize(x, p=2.0, dim=1)
        return self.layers(x)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

class NonVisualExtractor(nn.Module):
    def __init__(
            self,
            dim,
            object_size,
            temperature,
    ):
        super().__init__()
        self.dim = dim
        self.object_size = object_size
        self.temperature = temperature

        self.nv_extractor = nn.Linear(dim, object_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        pseudo_label = self.nv_extractor(x)
        pseudo_label = self.softmax(pseudo_label / self.temperature)
        return pseudo_label

class NonVisualEmbeddings(nn.Module):
    def __init__(
            self,
            dim,
            object_size,
            temperature,
    ):
        super().__init__()
        self.dim = dim
        self.object_size = object_size
        self.temperature = temperature

        self.nv_extractor = NonVisualExtractor(
            dim=dim,
            object_size=object_size,
            temperature=temperature,
        )
        if object_size < dim:
            self.nv_embeddings = nn.Parameter(torch.eye(dim)[:object_size, :],
                                              requires_grad=True)
        else:
            self.nv_embeddings = nn.Parameter(torch.eye(object_size)[:, :dim],
                                              requires_grad=True)

        self.cls_token_nv =  nn.Parameter(torch.zeros(1, 1, dim))

        nn.init.xavier_normal_(self.nv_embeddings)

    def forward(self, x):
        n = x.shape[1]
        cls, patches = torch.split(x, [1, n - 1], dim=1)
        batch_size = cls.shape[0]

        pseudo_label = self.nv_extractor(patches)
        nv_hidden_states = torch.matmul(pseudo_label, self.nv_embeddings)
        cls_token_nv = self.cls_token_nv.expand(batch_size, -1, -1)
        x = torch.cat([cls_token_nv, nv_hidden_states], dim=1)
        return x, pseudo_label

class NVEncoderBlock(nn.Module):
    """Transformer encoder block.
    From @myleott -
    There are at least three common structures.
    1) Attention is all you need had the worst one, where the layernorm came after each
        block and was in the residual path.
    2) BERT improved upon this by moving the layernorm to the beginning of each block
        (and adding an extra layernorm at the end).
    3) There's a further improved version that also moves the layernorm outside of the
        residual path, which is what this implementation does.
    Figure 1 of this paper compares versions 1 and 3:
        https://openreview.net/pdf?id=B1x8anVFPr
    Figure 7 of this paper compares versions 2 and 3 for BERT:
        https://arxiv.org/abs/1909.08053
    """

    def __init__(
        self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate, object_size, temperature
    ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout_rate
        )  # uses correct initialization by default
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)
        self.num_heads = num_heads

        self.nv_cluster = NonVisualEmbeddings(
            dim=hidden_dim,
            object_size=object_size,
            temperature=temperature,
        )

    def forward(self, hidden):
        input, pseudo_label, intermediates = hidden
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        x, pseudo_label = self.nv_cluster(x)
        intermediates.append(x.clone())
        return (x, pseudo_label, intermediates)

    def flops(self, x):
        flops = 0
        seq_len, batch_size, hidden_dim = x.shape

        num_elems = x.numel() // batch_size
        flops += num_elems * 6  # ln_1 (* 2), x + input, ln_2 (* 2), x + y

        # self_attention
        # calculations are based on the fact that head_dim * num_heads = hidden_dim
        # so we collapse (hidden_dim // num_heads) * num_heads to hidden_dim
        flops += 3 * seq_len * (hidden_dim + 1) * hidden_dim  # projection with bias
        flops += hidden_dim * seq_len  # scaling
        flops += hidden_dim * seq_len * seq_len  # attention weights
        flops += self.num_heads * seq_len * seq_len  # softmax
        flops += hidden_dim * seq_len * seq_len  # attention application
        flops += seq_len * (hidden_dim + 1) * hidden_dim  # out projection with bias

        # mlp
        mlp_dim = self.mlp.linear_1.out_features
        flops += seq_len * (hidden_dim + 1) * mlp_dim  # linear_1
        flops += seq_len * mlp_dim  # act
        flops += seq_len * (mlp_dim + 1) * hidden_dim  # linear_2
        return flops * batch_size

    def activations(self, out, x):
        # we only count activations for matrix multiplications
        activations = 0
        seq_len, batch_size, hidden_dim = x.shape

        # self_attention
        activations += 3 * seq_len * hidden_dim  # projection
        activations += self.num_heads * seq_len * seq_len  # attention weights
        activations += hidden_dim * seq_len  # attention application
        activations += hidden_dim * seq_len  # out projection

        # mlp
        mlp_dim = self.mlp.linear_1.out_features
        activations += seq_len * mlp_dim  # linear_1
        activations += seq_len * hidden_dim  # linear_2
        return activations

class EncoderBlock(nn.Module):
    """Transformer encoder block.
    From @myleott -
    There are at least three common structures.
    1) Attention is all you need had the worst one, where the layernorm came after each
        block and was in the residual path.
    2) BERT improved upon this by moving the layernorm to the beginning of each block
        (and adding an extra layernorm at the end).
    3) There's a further improved version that also moves the layernorm outside of the
        residual path, which is what this implementation does.
    Figure 1 of this paper compares versions 1 and 3:
        https://openreview.net/pdf?id=B1x8anVFPr
    Figure 7 of this paper compares versions 2 and 3 for BERT:
        https://arxiv.org/abs/1909.08053
    """

    def __init__(
        self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate
    ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout_rate
        )  # uses correct initialization by default
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)
        self.num_heads = num_heads

    def forward(self, hidden):
        input, pseudo_label, intermediates = hidden
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        intermediates.append(x.clone())
        return (x, pseudo_label, intermediates)

    def flops(self, x):
        flops = 0
        seq_len, batch_size, hidden_dim = x.shape

        num_elems = x.numel() // batch_size
        flops += num_elems * 6  # ln_1 (* 2), x + input, ln_2 (* 2), x + y

        # self_attention
        # calculations are based on the fact that head_dim * num_heads = hidden_dim
        # so we collapse (hidden_dim // num_heads) * num_heads to hidden_dim
        flops += 3 * seq_len * (hidden_dim + 1) * hidden_dim  # projection with bias
        flops += hidden_dim * seq_len  # scaling
        flops += hidden_dim * seq_len * seq_len  # attention weights
        flops += self.num_heads * seq_len * seq_len  # softmax
        flops += hidden_dim * seq_len * seq_len  # attention application
        flops += seq_len * (hidden_dim + 1) * hidden_dim  # out projection with bias

        # mlp
        mlp_dim = self.mlp.linear_1.out_features
        flops += seq_len * (hidden_dim + 1) * mlp_dim  # linear_1
        flops += seq_len * mlp_dim  # act
        flops += seq_len * (mlp_dim + 1) * hidden_dim  # linear_2
        return flops * batch_size

    def activations(self, out, x):
        # we only count activations for matrix multiplications
        activations = 0
        seq_len, batch_size, hidden_dim = x.shape

        # self_attention
        activations += 3 * seq_len * hidden_dim  # projection
        activations += self.num_heads * seq_len * seq_len  # attention weights
        activations += hidden_dim * seq_len  # attention application
        activations += hidden_dim * seq_len  # out projection

        # mlp
        mlp_dim = self.mlp.linear_1.out_features
        activations += seq_len * mlp_dim  # linear_1
        activations += seq_len * hidden_dim  # linear_2
        return activations


class Encoder(nn.Module):
    """Transformer Encoder."""

    def __init__(
        self,
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate,
        attention_dropout_rate,
        num_nvlayers,
        object_size,
        temperature,
        cluster_layer,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02)
        )
        self.cluster_layer = cluster_layer
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        for i in range(num_layers):
            layers.append(
                (
                    f"layer_{i}",
                    EncoderBlock(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                    ),
                )
            )
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

        nv_layers = []
        nv_layers.append(
            (
                f"nvlayer_{0}",
                NVEncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout_rate,
                    attention_dropout_rate,
                    object_size,
                    temperature,
                ),
            )
        )
        for i in range(1,num_nvlayers):
            nv_layers.append(
                (
                    f"layer_{i}",
                    EncoderBlock(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                    ),
            )
        )
        self.nv_layers = nn.Sequential(OrderedDict(nv_layers))
        self.nv_ln = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.pos_embedding  # should broadcast to the same shape
        v_x, _, intermediates = self.layers((self.dropout(x), None, []))

        nv_x, pseudo_label, intermediates = self.nv_layers((intermediates[self.cluster_layer-1], None, intermediates))

        v_x = self.ln(v_x)
        nv_x = self.nv_ln(nv_x)
        return v_x, nv_x, pseudo_label


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        num_nvlayers,
        object_size,
        temperature,
        cluster_layer,
        alpha,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        conv_stem_layers: Union[List[ConvStemLayer], List[Dict], None] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Input shape indivisible by patch size"
        assert classifier in ["token", "gap"], "Unexpected classifier mode"
        assert num_classes is None or is_pos_int(num_classes)

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.classifier = classifier
        self.alpha = alpha
        self.num_nvlayers = num_nvlayers
        self.object_size = object_size
        self.temperature = temperature
        self.cluster_layer = cluster_layer

        input_channels = 3

        self.conv_stem_layers = conv_stem_layers
        if conv_stem_layers is None:
            # conv_proj is a more efficient version of reshaping, permuting and projecting
            # the input
            self.conv_proj = nn.Conv2d(
                input_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            prev_channels = input_channels
            self.conv_proj = nn.Sequential()
            for i, conv_stem_layer in enumerate(conv_stem_layers):
                if isinstance(conv_stem_layer, Mapping):
                    conv_stem_layer = ConvStemLayer(**conv_stem_layer)
                kernel = conv_stem_layer.kernel
                stride = conv_stem_layer.stride
                out_channels = conv_stem_layer.out_channels
                padding = get_same_padding_for_kernel_size(kernel)
                self.conv_proj.add_module(
                    f"conv_{i}",
                    nn.Conv2d(
                        prev_channels,
                        out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                )
                self.conv_proj.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels))
                self.conv_proj.add_module(f"relu_{i}", nn.ReLU())
                prev_channels = out_channels
            self.conv_proj.add_module(
                f"conv_{i + 1}", nn.Conv2d(prev_channels, hidden_dim, kernel_size=1)
            )

        seq_length = (image_size // patch_size) ** 2
        if self.classifier == "token":
            # add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
            num_nvlayers,
            object_size,
            temperature,
            cluster_layer,
        )
        self.trunk_output = nn.Identity()
        self.nv_trunk_output = nn.Identity()

        self.seq_length = seq_length
        self.init_weights()

        if num_classes is not None:
            self.head = VisionTransformerHead(
                num_classes=num_classes, in_plane=hidden_dim
            )
            self.nv_head = VisionTransformerHead(
                num_classes=num_classes, in_plane=hidden_dim
            )
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.head = None

    def init_weights(self):
        if self.conv_stem_layers is None:
            lecun_normal_init(
                self.conv_proj.weight,
                fan_in=self.conv_proj.in_channels
                * self.conv_proj.kernel_size[0]
                * self.conv_proj.kernel_size[1],
            )
            nn.init.zeros_(self.conv_proj.bias)

    def diversity_loss(self, pseudo_label):
        entropy = pseudo_label.mean(dim=(0, 1))
        entropy = (entropy + 1e-12) * torch.log(entropy + 1e-12) * (-1)
        entropy = entropy.sum()

        return entropy

    def loss(self, ce_loss, div_loss):
        return ce_loss - 0.5 * div_loss

    def confusion_matrix(self, x_v, x_nv, labels):
        _, v_pred = torch.topk(x_v.clone().detach(), 1, dim=-1)
        _, nv_pred = torch.topk(x_nv.clone().detach(), 1, dim=-1)

        v_correct = v_pred.eq(labels.unsqueeze(1).expand_as(v_pred)).squeeze(1)
        nv_correct = nv_pred.eq(labels.unsqueeze(1).expand_as(nv_pred)).squeeze(1)

        TT_position = torch.mul((v_correct == True).to(torch.uint8), (nv_correct == True).to(torch.uint8)).bool()
        TF_position = torch.mul((v_correct == True).to(torch.uint8), (nv_correct == False).to(torch.uint8)).bool()
        FT_position = torch.mul((v_correct == False).to(torch.uint8), (nv_correct == True).to(torch.uint8)).bool()
        FF_position = torch.mul((v_correct == False).to(torch.uint8), (nv_correct == False).to(torch.uint8)).bool()

        return (TT_position, TF_position, FT_position, FF_position)

    def forward(self, x: torch.Tensor, labels: torch.Tensor): #labels: torch.Tensor
        assert x.ndim == 4, "Unexpected input shape"
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == w == self.image_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> ((n_h * n_w), n, hidden_dim)
        # the self attention layer expects inputs in the format (S, N, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(2, 0, 1)

        if self.classifier == "token":
            # expand the class token to the full batch
            batch_class_token = self.class_token.expand(-1, n, -1)
            x = torch.cat([batch_class_token, x], dim=0)

        x_v, x_nv, pseudo_label = self.encoder(x)

        if self.classifier == "token":
            # just return the output for the class token
            x_v = x_v[0, :, :]
            x_nv = x_nv[0, :, :]
        else:
            x_v = x_v.mean(dim=0)
            x_nv = x_nv.mean(dim=0)

        x_v = self.trunk_output(x_v)
        x_nv = self.nv_trunk_output(x_nv)
        if self.head is not None:
            x_v = self.head(x_v)
            x_nv = self.nv_head(x_nv)
            # loss = self.criterion(x, labels)
        else:
            pass
            # loss = None

        if self.training:
            logits = x_nv
        else:
            x_v = self.softmax(x_v)
            x_nv = self.softmax(x_nv)
            x_v = self.alpha * x_v
            x_nv = (1 - self.alpha) * x_nv
            logits = x_v + x_nv

        ce_loss = self.criterion(logits, labels)
        div_loss = self.diversity_loss(pseudo_label)
        loss = self.loss(ce_loss, div_loss)

        # confusion_position = self.confusion_matrix(x_v, x_nv, labels)

        return logits, loss, ce_loss.item(), div_loss.item()

    def load_state_dict(self, state, strict=True):
        # shape of pos_embedding is (seq_length, 1, hidden_dim)
        pos_embedding = state["encoder.pos_embedding"]
        seq_length, n, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(
                f"Unexpected position embedding shape: {pos_embedding.shape}"
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Position embedding hidden_dim incorrect: {hidden_dim}"
                f", expected: {self.hidden_dim}"
            )
        new_seq_length = self.seq_length

        if new_seq_length != seq_length:
            # need to interpolate the weights for the position embedding
            # we do this by reshaping the positions embeddings to a 2d grid, performing
            # an interpolation in the (h, w) space and then reshaping back to a 1d grid
            if self.classifier == "token":
                # the class token embedding shouldn't be interpolated so we split it up
                seq_length -= 1
                new_seq_length -= 1
                pos_embedding_token = pos_embedding[:1, :, :]
                pos_embedding_img = pos_embedding[1:, :, :]
            else:
                pos_embedding_token = pos_embedding[:0, :, :]  # empty data
                pos_embedding_img = pos_embedding
            # (seq_length, 1, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(1, 2, 0)
            seq_length_1d = int(math.sqrt(seq_length))
            assert (
                seq_length_1d * seq_length_1d == seq_length
            ), "seq_length is not a perfect square"

            logging.info(
                "Interpolating the position embeddings from image "
                f"{seq_length_1d * self.patch_size} to size {self.image_size}"
            )

            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(
                1, hidden_dim, seq_length_1d, seq_length_1d
            )
            new_seq_length_1d = self.image_size // self.patch_size

            # use bicubic interpolation - it gives significantly better results in
            # the test `test_resolution_change`
            new_pos_embedding_img = torch.nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode="bicubic",
                align_corners=True,
            )

            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_l)
            new_pos_embedding_img = new_pos_embedding_img.reshape(
                1, hidden_dim, new_seq_length
            )
            # (1, hidden_dim, new_seq_length) -> (new_seq_length, 1, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(2, 0, 1)
            new_pos_embedding = torch.cat(
                [pos_embedding_token, new_pos_embedding_img], dim=0
            )
            state["encoder.pos_embedding"] = new_pos_embedding
        super().load_state_dict(state, strict=strict)


class ViTB16(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        num_nvlayers=6,
        object_size=2048,
        temperature=0.1,
        cluster_layer=11,
        alpha=0.5,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
            num_classes=num_classes,
            num_nvlayers=num_nvlayers,
            object_size=object_size,
            temperature=temperature,
            cluster_layer=cluster_layer,
            alpha=alpha,
        )


class ViTL16(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        num_nvlayers=6,
        object_size=2048,
        temperature=0.1,
        cluster_layer=11,
        alpha=0.5,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
            num_classes=num_classes,
            num_nvlayers=num_nvlayers,
            object_size=object_size,
            temperature=temperature,
            cluster_layer=cluster_layer,
            alpha=alpha,
        )


class ViTH14(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
            num_classes=num_classes,
            num_nvlayers=6,
            object_size=2048,
            temperature=0.1,
            cluster_layer=31,
            alpha=0.5,
        )
