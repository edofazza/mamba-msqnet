import time
import math
import torch
from torch import nn, Tensor
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from utils.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.classification import MultilabelConfusionMatrix
import numpy as np

from transformers import TimesformerModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging
from .videomamba import videomamba_middle, videomamba_small, videomamba_tiny
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns


from mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from functools import partial


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class VideoMambaCLIPInitVideoGuideMultiLayerMamba(nn.Module):

    def __init__(self, class_embed, num_frames, version: str = 'm'):
        super().__init__()
        assert version in ['m', 's', 't'], 'version must be m or s or t'
        self.num_classes, self.embed_dim = class_embed.shape
        #print(class_embed.shape, flush=True)
        if version == 'm':
            self.backbone = videomamba_middle(num_frames=num_frames, pretrained=True)
        elif version == 's':
            self.backbone = videomamba_small(num_frames=num_frames, pretrained=True)
        elif version == 't':
            self.backbone = videomamba_tiny(num_frames=num_frames, pretrained=True)

        if version == 'm':  # self.backbone.config.hidden_size # M 576 S 384 T 192
            self.linear1 = nn.Linear(in_features=576, out_features=self.embed_dim, bias=False)
        elif version == 's':
            self.linear1 = nn.Linear(in_features=384, out_features=self.embed_dim, bias=False)
        elif version == 't':
            self.linear1 = nn.Linear(in_features=192, out_features=self.embed_dim, bias=False)

        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.linear2 = nn.Linear(in_features=768 + self.embed_dim,
                                 out_features=self.embed_dim, bias=False)
        self.query_embed = nn.Parameter(class_embed)
        self.layers = nn.ModuleList(
            [
                create_block(
                    self.embed_dim,
                    layer_idx=i,
                )
                for i in range(16)
            ]
        )
        self.linear3 = nn.Linear(in_features=156, out_features=self.num_classes, bias=False)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone.forward_features(images.reshape(b, c, t, h, w))
        #print('POST BACKBONE:', x.size(), flush=True)
        x = self.linear1(F.adaptive_avg_pool1d(x.transpose(1, 2), 16).transpose(1, 2))
        #print('POST LINEAR1:', x.size(), flush=True)
        x = self.pos_encod(x)
        #print('POST ENCODER:', x.size(), flush=True)
        video_features = self.image_model(images.reshape(b * t, c, h, w))[1].reshape(b, t, -1).mean(dim=1, keepdim=True)
        #print('POST IMAGE MODEL:', video_features.size(), flush=True)
        #print('INPUT LINEAR2:', torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), video_features.repeat(1, self.num_classes, 1)),
        #                 2).size(), flush=True )
        query_embed = self.linear2(
            torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), video_features.repeat(1, self.num_classes, 1)),
                         2))
        x = torch.cat((x, query_embed), dim=1)
        #print('POS CONCAT:', x.size(), flush=True)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        #print('POS MAMBA:', x.size(), flush=True)
        _, d1, d2 = x.size()
        x = self.linear3(x.reshape(b, d2, d1)).reshape(b, self.num_classes, -1)
        """print('AFTER MAMBA', x.size(), flush=True)
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), 140).transpose(1, 2)
        print(x.size(), flush=True)
        #x = self.linear3(x.reshape(b, d2, d1)).reshape(b, self.num_classes, -1)"""
        out = self.group_linear(x)
        return out


class VideoMambaCLIPInitVideoGuideMultiLayerMambaExecutor:

    def __init__(self, train_loader, test_loader, criterion, eval_metric, class_list, test_every, distributed,
                 gpu_id, version: str = 'm') -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metric = eval_metric.to(gpu_id)
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        num_frames = self.train_loader.dataset[0][0].shape[0]
        logging.set_verbosity_error()
        class_embed = self._get_text_features(class_list)
        model = VideoMambaCLIPInitVideoGuideMultiLayerMamba(class_embed, num_frames, version).to(gpu_id)
        if distributed:
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.image_model.parameters():
            p.requires_grad = False
        #self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.0001}])  # 1e-5
        self.optimizer = AdamW(self.model.parameters(),  lr=0.0001, betas=(0.9, 0.95), weight_decay=0.1)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    @staticmethod
    def _get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt

    def _get_text_features(self, cl_names):
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        act_prompt = self._get_prompt(cl_names)
        texts = tokenizer(act_prompt, padding=True, return_tensors="pt")
        text_class = text_model(**texts).pooler_output.detach()
        return text_class

    def _train_batch(self, data, label):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss_this = self.criterion(output, label)
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label in self.train_loader:
            data, label = data.to(self.gpu_id, non_blocking=True), label.to(self.gpu_id, non_blocking=True)
            loss_this = self._train_batch(data, label)
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        self.scheduler.step()
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg), flush=True)

    def train(self, start_epoch, end_epoch):
        print('FROZEN BACKBONE', flush=True)
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        for name, param in self.model.backbone.named_parameters():
            print(name, param.requires_grad)
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
        print('UNFROZEN BACKBONE', flush=True)
        #self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00001}])
        #self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        for name, param in self.model.backbone.named_parameters():
            print(name, param.requires_grad)
        for epoch in range(start_epoch, int(end_epoch * 1.5)):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)


    def test(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, label in self.test_loader:
            data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
            with torch.no_grad():
                output = self.model(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg

    def save(self, file_path="./checkpoint.pth"):
        torch.save(self.model.state_dict(), file_path + '.pth')
        torch.save(self.optimizer.state_dict(), file_path + '_optimizer.pth')

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path))


class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x