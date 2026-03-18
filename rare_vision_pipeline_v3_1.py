###############################################################################
# ICPR 2026 RARE-VISION — Differential BiomedCLIP Pipeline v3.1
# Modes: smoke | train | test
#
# v3.1: + Resume from checkpoint (set RESUME_FROM path + update EPOCHS)
#        + All v3 anti-overfitting: augmentation, mixup, EMA, label-smoothing
#        + Sqrt-freq sampling, per-class threshold, correct JSON
###############################################################################

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import subprocess, sys
def install_packages():
    for p in ["open_clip_torch","timm","pandas","scikit-learn",
              "matplotlib","seaborn","tqdm","openpyxl","scipy"]:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
install_packages()

import json, glob, copy, math, time, random, warnings, gc, logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import median_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score, roc_auc_score,
)
import timm
from timm.layers import DropPath, Mlp
from open_clip import create_model_from_pretrained, get_tokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    MODE = "smoke"  # "smoke" | "train" | "test"

    # ── RESUME ─────────────────────────────────────────────────────────────
    # Set this to your checkpoint path to continue training.
    RESUME_FROM = None 

    # ── Paths ──
    DATASET_ROOT    = "/media/data/Galar_Dataset/Galar_Dataset"
    LABELS_DIR      = os.path.join(DATASET_ROOT, "20251215_Labels_Updated")
    LABELS_DIR_ORIG = os.path.join(DATASET_ROOT, "Galar_labels_and_metadata", "Labels")
    FRAME_DIR_PATTERN = os.path.join(DATASET_ROOT, "Galar_Frames_*")
    TEST_DATA_ROOT  = "/home/satyajith/Downloads/Testdata_ICPR_2026_RARE_Challenge"
    OUTPUT_DIR      = "./rare_vision_output_v3"
    CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
    RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")
    LOGS_DIR        = os.path.join(OUTPUT_DIR, "logs")
    CURVES_DIR      = os.path.join(OUTPUT_DIR, "curves")

    # ── Model ──
    MODEL_NAME   = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    NUM_CLASSES  = 17; IMAGE_SIZE = 224; LAMBDA_INIT = 0.8; EB_REDUCTION = 16

    # ── Labels ──
    ANATOMICAL_LABELS  = ["mouth","esophagus","stomach","small intestine",
                          "colon","z-line","pylorus","ileocecal valve"]
    PATHOLOGICAL_LABELS= ["active bleeding","angiectasia","blood","erosion",
                          "erythema","hematin","lymphangioectasis","polyp","ulcer"]
    ALL_LABELS = ANATOMICAL_LABELS + PATHOLOGICAL_LABELS
    ALL_LABELS_INTERNAL = [l.replace(" ","_") for l in ALL_LABELS]
    LABEL_ALIASES = {}
    for _l in ALL_LABELS:
        _k = _l.replace(" ","_")
        for v in [_l, _l.replace(" ",""), _k, _l.replace("-","_")]:
            LABEL_ALIASES[v.lower()] = _k

    # ── Training ──
    BATCH_SIZE      = 128
    NUM_WORKERS     = 8
    EPOCHS          = 10
    LR              = 3e-4
    BACKBONE_LR_MULT= 0.3
    WEIGHT_DECAY    = 5e-4
    GRAD_ACCUM_STEPS= 1
    EARLY_STOP_PATIENCE = 15
    USE_AMP         = True

    # ── Anti-overfitting ──
    LABEL_SMOOTHING = 0.05
    DROPOUT         = 0.4
    USE_MIXUP       = True; MIXUP_ALPHA = 0.3
    USE_EMA         = True; EMA_DECAY = 0.999
    USE_STRONG_AUG  = True

    # ── Class imbalance ──
    USE_FOCAL_LOSS  = True; FOCAL_GAMMA_POS = 1; FOCAL_GAMMA_NEG = 4
    USE_CONTRASTIVE = True; CONTRASTIVE_WEIGHT = 0.4; CLS_WEIGHT = 1.0
    USE_CLASS_BALANCED_SAMPLING = True

    # ── Temporal ──
    MIN_EVENT_LENGTH = 3; SMOOTH_WINDOW = 7; CONFIDENCE_THRESH = 0.5
    RARE_CLASS_THRESH = 0.2; MERGE_GAP = 5

    # ── Sampling ──
    FRAME_SAMPLE_RATE = 5; VAL_SAMPLE_RATE = 5; VAL_SPLIT = 0.15

    # ── Smoke ──
    SMOKE_NUM_VIDEOS=2; SMOKE_EPOCHS=2; SMOKE_BATCH_SIZE=4
    SMOKE_MAX_FRAMES=200; SMOKE_SAMPLE_RATE=20

    SEED = 42

    @classmethod
    def apply_mode(cls):
        if cls.MODE == "smoke":
            cls.EPOCHS=cls.SMOKE_EPOCHS; cls.BATCH_SIZE=cls.SMOKE_BATCH_SIZE
            cls.FRAME_SAMPLE_RATE=cls.SMOKE_SAMPLE_RATE
            cls.VAL_SAMPLE_RATE=cls.SMOKE_SAMPLE_RATE; cls.NUM_WORKERS=0
            cls.USE_AMP=False; cls.USE_CLASS_BALANCED_SAMPLING=False
            cls.USE_MIXUP=False; cls.USE_EMA=False; cls.USE_STRONG_AUG=False
            cls.RESUME_FROM=None
            logger.info("🔬 SMOKE MODE")
        elif cls.MODE == "train": logger.info("🚀 TRAIN MODE")
        elif cls.MODE == "test": logger.info("🧪 TEST MODE")
        for d in [cls.OUTPUT_DIR,cls.CHECKPOINT_DIR,cls.RESULTS_DIR,cls.LOGS_DIR,cls.CURVES_DIR]:
            os.makedirs(d, exist_ok=True)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=False; torch.backends.cudnn.benchmark=True

# ============================================================================
# DATA
# ============================================================================
def canonicalize_label(name):
    key = name.strip().lower()
    if key in Config.LABEL_ALIASES: return Config.LABEL_ALIASES[key]
    key2 = key.replace(" ","_")
    if key2 in Config.LABEL_ALIASES: return Config.LABEL_ALIASES[key2]
    return None

def load_video_labels(video_id):
    csv_path = os.path.join(Config.LABELS_DIR, f"{video_id}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(Config.LABELS_DIR_ORIG, f"{video_id}.csv")
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    frame_col = None
    for col in df.columns:
        if col.strip().lower() in ("frame","frame_number","frame_no","index"):
            frame_col = col; break
    if frame_col is None: frame_col = df.columns[-1]
    rename_map = {frame_col: "frame_number"}
    for col in df.columns:
        if col == frame_col: continue
        canon = canonicalize_label(col)
        if canon and canon in Config.ALL_LABELS_INTERNAL: rename_map[col] = canon
    df = df.rename(columns=rename_map)
    for label in Config.ALL_LABELS_INTERNAL:
        if label not in df.columns: df[label] = 0
    for col in Config.ALL_LABELS_INTERNAL:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(["1","true","yes"]).astype(np.int8)
        else: df[col] = df[col].fillna(0).astype(np.int8)
    df["frame_number"] = df["frame_number"].astype(int)
    return df.set_index("frame_number").sort_index()

def discover_all_videos():
    videos = {}
    for rf in sorted(glob.glob(Config.FRAME_DIR_PATTERN)):
        if not os.path.isdir(rf): continue
        for vf in sorted(os.listdir(rf)):
            vp = os.path.join(rf, vf)
            if os.path.isdir(vp):
                try: videos[int(vf)] = vp
                except ValueError: pass
    logger.info(f"Discovered {len(videos)} videos")
    return videos

def discover_test_videos():
    videos = {}
    root = Config.TEST_DATA_ROOT
    if not os.path.isdir(root): return videos
    for item in sorted(os.listdir(root)):
        ip = os.path.join(root, item)
        if os.path.isdir(ip):
            sample = os.listdir(ip)[:5]
            if any(f.lower().endswith((".png",".jpg",".jpeg")) for f in sample):
                videos[item] = ip
    logger.info(f"Discovered {len(videos)} test videos")
    return videos

# ============================================================================
# AUGMENTATION
# ============================================================================
def build_train_transform(base_preprocess):
    if not Config.USE_STRONG_AUG: return base_preprocess
    return T.Compose([
        T.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.7,1.0), ratio=(0.9,1.1)),
        T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(0.05), T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711]),
        T.RandomErasing(p=0.2, scale=(0.02,0.15)),
    ])

# ============================================================================
# DATASET
# ============================================================================
class GalarVideoDataset(Dataset):
    def __init__(self, video_ids, video_dirs, transform=None, sample_rate=5,
                 max_frames_per_video=None, is_test=False):
        self.transform=transform; self.is_test=is_test; self.samples=[]
        for vid_id in tqdm(video_ids, desc="Loading video metadata"):
            frame_dir = video_dirs.get(vid_id)
            if frame_dir is None: continue
            if not is_test:
                labels_df = load_video_labels(vid_id)
                if labels_df is None or len(labels_df)==0: continue
                labeled_frames = set(labels_df.index.tolist())
            else: labels_df=None; labeled_frames=None
            frame_files = sorted([f for f in os.listdir(frame_dir)
                                  if f.lower().endswith((".png",".jpg",".jpeg"))])
            frame_items = []
            for ff in frame_files:
                try:
                    fnum = int(ff.split("_")[-1].split(".")[0])
                    frame_items.append((fnum, os.path.join(frame_dir, ff)))
                except: continue
            frame_items.sort(key=lambda x: x[0])
            sampled = frame_items[::sample_rate]
            if max_frames_per_video and len(sampled)>max_frames_per_video:
                sampled = sampled[:max_frames_per_video]
            for fnum, fpath in sampled:
                if is_test: lv = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
                else:
                    if fnum not in labeled_frames: continue
                    lv = labels_df.loc[fnum, Config.ALL_LABELS_INTERNAL].values.astype(np.float32)
                self.samples.append((fpath, lv, vid_id, fnum))
        logger.info(f"Dataset: {len(self.samples)} samples from {len(video_ids)} videos")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fpath, lv, vid_id, fnum = self.samples[idx]
        try:
            with Image.open(fpath) as tmp: img = tmp.convert("RGB")
        except: img = Image.new("RGB",(Config.IMAGE_SIZE,Config.IMAGE_SIZE))
        if self.transform: img = self.transform(img)
        return img, torch.tensor(lv, dtype=torch.float32), vid_id, fnum

class GalarSequentialVideoDataset(Dataset):
    def __init__(self, video_id, frame_dir, transform=None):
        self.video_id=video_id; self.transform=transform; self.frames=[]
        for ff in sorted(os.listdir(frame_dir)):
            if ff.lower().endswith((".png",".jpg",".jpeg")):
                try:
                    fnum = int(ff.split("_")[-1].split(".")[0])
                    self.frames.append((fnum, os.path.join(frame_dir, ff)))
                except: continue
        self.frames.sort(key=lambda x: x[0])
        logger.info(f"Video {video_id}: {len(self.frames)} frames")
    def __len__(self): return len(self.frames)
    def __getitem__(self, idx):
        fnum, fpath = self.frames[idx]
        try:
            with Image.open(fpath) as tmp: img = tmp.convert("RGB")
        except: img = Image.new("RGB",(Config.IMAGE_SIZE,Config.IMAGE_SIZE))
        if self.transform: img = self.transform(img)
        return img, fnum

# ============================================================================
# SAMPLER / MIXUP / EMA
# ============================================================================
def build_class_balanced_sampler(dataset):
    all_labels = np.stack([s[1] for s in dataset.samples], axis=0)
    class_counts = all_labels.sum(axis=0) + 1.0
    class_inv_sqrt = 1.0 / np.sqrt(class_counts / len(all_labels))
    weights = np.zeros(len(dataset), dtype=np.float64)
    for i in range(len(dataset)):
        active = np.where(all_labels[i]>0)[0]
        weights[i] = max(class_inv_sqrt[j] for j in active) if len(active)>0 else 1.0
    weights /= weights.sum(); weights *= len(dataset)
    logger.info(f"Sampler: [{weights.min():.3f}, {weights.max():.3f}]")
    return WeightedRandomSampler(weights, len(dataset), replacement=True)

def mixup_data(images, labels, alpha=0.3):
    if alpha <= 0: return images, labels
    lam = max(np.random.beta(alpha,alpha), 0.5)
    idx = torch.randperm(images.size(0), device=images.device)
    return lam*images+(1-lam)*images[idx], lam*labels+(1-lam)*labels[idx]

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay=decay; self.shadow={}; self.backup={}
        for n,p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.data.clone()
    @torch.no_grad()
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply_shadow(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n]=p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n,p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup={}

# ============================================================================
# MODEL
# ============================================================================
class DifferentialMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., lambda_init=0.8):
        super().__init__()
        assert num_heads%2==0
        self.num_heads=num_heads; self.effective_heads=num_heads//2
        self.head_dim=embed_dim//num_heads; self.scale=self.head_dim**-0.5
        self.q_proj=nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.k_proj=nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.v_proj=nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(embed_dim,embed_dim); self.proj_drop=nn.Dropout(proj_drop)
        self.lambda_q1=nn.Parameter(torch.zeros(self.effective_heads,1,self.head_dim))
        self.lambda_k1=nn.Parameter(torch.zeros(self.effective_heads,1,self.head_dim))
        self.lambda_q2=nn.Parameter(torch.zeros(self.effective_heads,1,self.head_dim))
        self.lambda_k2=nn.Parameter(torch.zeros(self.effective_heads,1,self.head_dim))
        self.lambda_init=lambda_init
        for p in [self.lambda_q1,self.lambda_k1,self.lambda_q2,self.lambda_k2]:
            nn.init.normal_(p, std=0.02)
    def forward(self, x):
        B,N,C=x.shape
        q=self.q_proj(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k=self.k_proj(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        v=self.v_proj(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        q1,q2=torch.chunk(q,2,dim=1); k1,k2=torch.chunk(k,2,dim=1); v1,v2=torch.chunk(v,2,dim=1)
        l1=torch.exp(torch.clamp((self.lambda_q1*self.lambda_k1).sum(-1).sum(-1),max=5))
        l2=torch.exp(torch.clamp((self.lambda_q2*self.lambda_k2).sum(-1).sum(-1),max=5))
        lv=(l1-l2+self.lambda_init).mean()
        a1=self.attn_drop((q1@k1.transpose(-2,-1)*self.scale).softmax(-1))
        x1=(a1@v1).transpose(1,2).reshape(B,N,C//2)
        a2=self.attn_drop((q2@k2.transpose(-2,-1)*self.scale).softmax(-1))
        x2=(a2@v2).transpose(1,2).reshape(B,N,C//2)
        return self.proj_drop(self.proj(torch.cat([x1-lv*x2, x2], dim=-1)))

class DifferentialBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, lambda_init=0.8):
        super().__init__()
        self.norm1=norm_layer(dim)
        self.attn=DifferentialMultiheadAttention(dim,num_heads,qkv_bias=qkv_bias,
            attn_drop=attn_drop,proj_drop=drop,lambda_init=lambda_init)
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2=norm_layer(dim)
        self.mlp=Mlp(in_features=dim,hidden_features=int(dim*mlp_ratio),act_layer=act_layer,drop=drop)
    def forward(self, x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        return x+self.drop_path(self.mlp(self.norm2(x)))

class ExcitationBlock(nn.Module):
    def __init__(self, in_features, reduction=16):
        super().__init__()
        self.fc1=nn.Linear(in_features,in_features//reduction)
        self.gate=nn.Sequential(nn.Linear(in_features//reduction,in_features),nn.Sigmoid())
        self.bn=nn.BatchNorm1d(in_features)
    def forward(self, x):
        return self.bn(x * self.gate(F.relu(self.fc1(x))))

class DifferentialBiomedCLIP(nn.Module):
    def __init__(self, model_name, num_classes, device, lambda_init=0.8,
                 eb_reduction=16, class_names=None, use_contrastive=True):
        super().__init__()
        self.device=device; self.num_classes=num_classes; self.use_contrastive=use_contrastive
        logger.info("Loading BiomedCLIP...")
        base_model, self.preprocess = create_model_from_pretrained('hf-hub:'+model_name, device=device)
        self.tokenizer = get_tokenizer('hf-hub:'+model_name)
        logger.info("Replacing ViT with Differential Attention...")
        tw=base_model.visual
        vit=tw.trunk if hasattr(tw,'trunk') else next(tw.children())
        ed=getattr(vit,'embed_dim',768); nh=getattr(vit,'num_heads',12)
        dpr_rate=getattr(vit,'drop_path_rate',0.0)
        dr=getattr(vit,'drop_rate',0.0); adr=getattr(vit,'attn_drop_rate',0.0)
        depth=len(vit.blocks)
        dpr=[x.item() for x in torch.linspace(0,dpr_rate,depth)]
        new_blocks=nn.Sequential(*[DifferentialBlock(dim=ed,num_heads=nh,mlp_ratio=4.0,
            qkv_bias=True,drop=dr,attn_drop=adr,drop_path=dpr[i],
            norm_layer=type(vit.blocks[0].norm1),act_layer=nn.GELU,
            lambda_init=lambda_init) for i in range(depth)])
        logger.info("Transferring pretrained weights...")
        with torch.no_grad():
            for orig,diff in zip(vit.blocks,new_blocks):
                diff.attn.q_proj.weight.copy_(orig.attn.qkv.weight[:ed])
                diff.attn.k_proj.weight.copy_(orig.attn.qkv.weight[ed:2*ed])
                diff.attn.v_proj.weight.copy_(orig.attn.qkv.weight[2*ed:])
                if orig.attn.qkv.bias is not None:
                    diff.attn.q_proj.bias.copy_(orig.attn.qkv.bias[:ed])
                    diff.attn.k_proj.bias.copy_(orig.attn.qkv.bias[ed:2*ed])
                    diff.attn.v_proj.bias.copy_(orig.attn.qkv.bias[2*ed:])
                diff.attn.proj.weight.copy_(orig.attn.proj.weight)
                diff.attn.proj.bias.copy_(orig.attn.proj.bias)
                diff.mlp.fc1.weight.copy_(orig.mlp.fc1.weight)
                diff.mlp.fc1.bias.copy_(orig.mlp.fc1.bias)
                diff.mlp.fc2.weight.copy_(orig.mlp.fc2.weight)
                diff.mlp.fc2.bias.copy_(orig.mlp.fc2.bias)
                diff.norm1.weight.copy_(orig.norm1.weight)
                diff.norm1.bias.copy_(orig.norm1.bias)
                diff.norm2.weight.copy_(orig.norm2.weight)
                diff.norm2.bias.copy_(orig.norm2.bias)
        vit.blocks=new_blocks; self.model=base_model
        if self.use_contrastive:
            self.logit_scale=nn.Parameter(torch.ones([])*np.log(1/0.07))
            if class_names is None: class_names=Config.ALL_LABELS
            self.class_names=class_names
            prompts=[f"a video capsule endoscopy image showing {cn}" for cn in class_names]
            self.register_buffer('text_features',self._encode_text(prompts))
        self.eb_block=ExcitationBlock(512,reduction=eb_reduction)
        self.dropout=nn.Dropout(Config.DROPOUT)
        self.classification_head=nn.Linear(512,num_classes)
        self.to(device)
    def _encode_text(self, text_list):
        tokens=self.tokenizer(text_list).to(self.device)
        with torch.no_grad():
            f=self.model.encode_text(tokens); f=f/f.norm(dim=-1,keepdim=True)
        return f
    def forward(self, images, return_contrastive=False, return_features=False):
        image_features=self.model.encode_image(images, normalize=False)
        logits_con=None
        if self.use_contrastive and return_contrastive:
            img_n=image_features/(image_features.norm(dim=-1,keepdim=True)+1e-8)
            scale=torch.clamp(self.logit_scale,max=4.6052).exp()
            logits_con=img_n@self.text_features.T*scale
        x=self.eb_block(image_features.float()); x=self.dropout(x)
        logits_cls=self.classification_head(x)
        if return_contrastive and logits_con is not None:
            return logits_con, logits_cls
        if return_features: return logits_cls, image_features
        return logits_cls

# ============================================================================
# LOSSES
# ============================================================================
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, clip=0.05, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.gp=gamma_pos; self.gn=gamma_neg; self.clip=clip; self.pw=pos_weight; self.ls=label_smoothing
    def forward(self, logits, targets):
        if self.ls>0: targets=targets*(1-self.ls)+(1-targets)*self.ls
        probs=torch.sigmoid(logits).clamp(1e-6,1-1e-6)
        xs_pos=probs; xs_neg=(1-probs).clamp(1e-6)
        if self.clip and self.clip>0: xs_neg=(xs_neg+self.clip).clamp(max=1)
        loss=targets*torch.log(xs_pos)+(1-targets)*torch.log(xs_neg)
        if self.gn>0 or self.gp>0:
            pt=xs_pos*targets+xs_neg*(1-targets)
            gamma=self.gp*targets+self.gn*(1-targets)
            loss*=torch.pow(1-pt,gamma)
        if self.pw is not None:
            loss=loss*(targets*self.pw.to(logits.device)+(1-targets))
        return -loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, cls_fn, con_w=0.3, cls_w=1.0, pos_weight=None):
        super().__init__()
        self.cls_fn=cls_fn; self.con_w=con_w; self.cls_w=cls_w; self.pw=pos_weight
    def forward(self, logits_cls, targets, logits_con=None):
        loss=self.cls_w*self.cls_fn(logits_cls,targets)
        if logits_con is not None and self.con_w>0:
            pw=self.pw.to(logits_con.device) if self.pw is not None else None
            loss+=self.con_w*F.binary_cross_entropy_with_logits(logits_con,targets,pos_weight=pw)
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0,device=logits_cls.device,requires_grad=True)
        return loss

def compute_class_weights(dataset):
    all_labels=np.stack([s[1] for s in dataset.samples],axis=0)
    pos=all_labels.sum(axis=0)+1; neg=len(all_labels)-pos+1
    pw=np.clip(neg/pos,1.0,100.0)
    logger.info("Class weights:")
    for i,l in enumerate(Config.ALL_LABELS):
        logger.info(f"  {l:25s}: {pw[i]:7.2f} (pos={int(pos[i]-1):>7})")
    return torch.tensor(pw,dtype=torch.float32)

# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(labels, probs, preds, names, save_dir=None, prefix="val"):
    n=len(names); pc=[]
    for c in range(n):
        m={"label":names[c],"support":int(labels[:,c].sum())}
        if m["support"]>0 and m["support"]<len(labels):
            m["AP"]=average_precision_score(labels[:,c],probs[:,c])
            m["AUC"]=roc_auc_score(labels[:,c],probs[:,c])
        else: m["AP"]=0.0; m["AUC"]=0.0
        m["precision"]=precision_score(labels[:,c],preds[:,c],zero_division=0)
        m["recall"]=recall_score(labels[:,c],preds[:,c],zero_division=0)
        m["f1"]=f1_score(labels[:,c],preds[:,c],zero_division=0)
        m["accuracy"]=float((preds[:,c]==labels[:,c]).sum())/len(labels)
        pc.append(m)
    ov={k:np.mean([m[k] for m in pc]) for k in ["AP","AUC","f1","precision","recall","accuracy"]}
    logger.info(f"\n{'='*90}")
    logger.info(f"{'Label':<25} {'AP':>7} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7} {'Sup':>8}")
    logger.info("-"*90)
    for m in pc:
        logger.info(f"{m['label']:<25} {m['AP']:>7.4f} {m['AUC']:>7.4f} {m['f1']:>7.4f} "
                     f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['accuracy']:>7.4f} {m['support']:>8}")
    logger.info("-"*90)
    logger.info(f"{'MACRO':<25} {ov['AP']:>7.4f} {ov['AUC']:>7.4f} {ov['f1']:>7.4f} "
                f"{ov['precision']:>7.4f} {ov['recall']:>7.4f} {ov['accuracy']:>7.4f}")
    logger.info("="*90)
    if save_dir:
        pd.DataFrame(pc).to_csv(os.path.join(save_dir,f"{prefix}_metrics.csv"),index=False)
        # PR curves
        fig,axes=plt.subplots(3,6,figsize=(30,15)); axes=axes.flatten()
        for c in range(n):
            ax=axes[c]
            if labels[:,c].sum()>0:
                pr,re,_=precision_recall_curve(labels[:,c],probs[:,c])
                ax.plot(re,pr,'b-',lw=1.5); ax.fill_between(re,pr,alpha=0.2)
                ax.set_title(f"{names[c]}\nAP={pc[c]['AP']:.3f}",fontsize=9)
            else: ax.set_title(f"{names[c]}\n(no pos)",fontsize=9)
            ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(True,alpha=0.3)
        if n<len(axes): axes[-1].axis('off')
        plt.suptitle(f"PR Curves ({prefix})"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir,f"{prefix}_pr.png"),dpi=150); plt.close()
        # ROC curves
        fig,axes=plt.subplots(3,6,figsize=(30,15)); axes=axes.flatten()
        for c in range(n):
            ax=axes[c]
            if labels[:,c].sum()>0 and labels[:,c].sum()<len(labels):
                fpr,tpr,_=roc_curve(labels[:,c],probs[:,c])
                ax.plot(fpr,tpr,'r-',lw=1.5); ax.plot([0,1],[0,1],'k--',alpha=0.3)
                ax.set_title(f"{names[c]}\nAUC={pc[c]['AUC']:.3f}",fontsize=9)
            else: ax.set_title(f"{names[c]}\n(n/a)",fontsize=9)
            ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(True,alpha=0.3)
        if n<len(axes): axes[-1].axis('off')
        plt.suptitle(f"ROC Curves ({prefix})"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir,f"{prefix}_roc.png"),dpi=150); plt.close()
        # Bar chart
        fig,axes=plt.subplots(1,3,figsize=(24,7)); x=np.arange(n); w=0.35
        axes[0].bar(x,[m["AP"] for m in pc],color='steelblue')
        axes[0].set_xticks(x); axes[0].set_xticklabels(names,rotation=45,ha='right',fontsize=7)
        axes[0].set_title("AP"); axes[0].set_ylim(0,1)
        axes[1].bar(x-w/2,[m["precision"] for m in pc],w,label='Prec',color='teal')
        axes[1].bar(x+w/2,[m["recall"] for m in pc],w,label='Rec',color='coral')
        axes[1].set_xticks(x); axes[1].set_xticklabels(names,rotation=45,ha='right',fontsize=7)
        axes[1].legend(); axes[1].set_ylim(0,1)
        axes[2].bar(x,[m["f1"] for m in pc],color='mediumpurple')
        axes[2].set_xticks(x); axes[2].set_xticklabels(names,rotation=45,ha='right',fontsize=7)
        axes[2].set_ylim(0,1)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"{prefix}_bar.png"),dpi=150); plt.close()
    return {"per_class":pc,"overall":ov}

def optimize_thresholds(labels, probs, names):
    total=len(labels); thresholds=np.full(len(names),0.5)
    for c in range(len(names)):
        sup=labels[:,c].sum()
        if sup==0: thresholds[c]=Config.RARE_CLASS_THRESH; continue
        best_f1=0; best_t=0.5
        lo=0.05 if sup<total*0.01 else 0.15
        for t in np.arange(lo,0.85,0.02):
            f=f1_score(labels[:,c],(probs[:,c]>=t).astype(int),zero_division=0)
            if f>best_f1: best_f1=f; best_t=t
        thresholds[c]=best_t
        tag=" (RARE)" if sup<total*0.01 else ""
        logger.info(f"  {names[c]:25s}: t={best_t:.2f} F1={best_f1:.4f} sup={int(sup)}{tag}")
    return thresholds

# ============================================================================
# JSON (matching make_json.py)
# ============================================================================
def frame_preds_to_events_grouped(frame_numbers, binary_preds, label_names):
    n=len(frame_numbers)
    if n==0: return []
    def active_set(i):
        return tuple(sorted(label_names[c] for c in range(len(label_names)) if binary_preds[i,c]==1))
    events=[]; cur=active_set(0); start=int(frame_numbers[0])
    for i in range(1,n):
        s=active_set(i)
        if s!=cur:
            events.append({"start":start,"end":int(frame_numbers[i])-1,"label":list(cur)})
            start=int(frame_numbers[i]); cur=s
    events.append({"start":start,"end":int(frame_numbers[-1]),"label":list(cur)})
    return events

def build_submission_json(all_events, path):
    sub={"videos":[{"video_id":str(vid),"events":evts}
                    for vid,evts in sorted(all_events.items(),key=lambda x:str(x[0]))]}
    with open(path,"w") as f: json.dump(sub,f,indent=2)
    logger.info(f"JSON → {path}"); return sub

# ============================================================================
# TRAINER (with resume)
# ============================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, cfg):
        self.model=model; self.tl=train_loader; self.vl=val_loader
        self.crit=criterion; self.opt=optimizer; self.sched=scheduler
        self.dev=device; self.cfg=cfg
        self.scaler=GradScaler() if cfg.USE_AMP else None
        self.ema=ModelEMA(model,cfg.EMA_DECAY) if cfg.USE_EMA else None
        self.best_map=0.0; self.pat=0
        self.hist={"tl":[],"vl":[],"map":[],"lr":[]}
        self.start_epoch=0

    def resume_from_checkpoint(self, ckpt_path):
        """Load checkpoint and continue training from saved epoch."""
        ckpt=torch.load(ckpt_path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["optimizer_state_dict"])
                logger.info("Optimizer state restored")
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")
        if "scaler_state_dict" in ckpt and self.scaler:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                logger.info("AMP scaler state restored")
            except: pass
        if "ema_shadow" in ckpt and self.ema:
            self.ema.shadow = ckpt["ema_shadow"]
            logger.info("EMA shadow weights restored")
        if "history" in ckpt:
            self.hist = ckpt["history"]
            # Ensure all keys exist (backward compat)
            for k in ["tl","vl","map","lr"]:
                if k not in self.hist: self.hist[k] = []
            logger.info(f"Training history restored ({len(self.hist['tl'])} epochs)")

        self.start_epoch = ckpt.get("epoch", 0) + 1  # +1 because saved epoch is completed
        self.best_map = ckpt.get("val_mAP", 0.0)

        # Advance scheduler to correct position
        if self.sched is not None:
            steps_done = self.start_epoch * len(self.tl)
            logger.info(f"Advancing scheduler by {steps_done} steps...")
            for _ in range(steps_done):
                self.sched.step()

        logger.info(f"✅ Resumed from epoch {self.start_epoch} (best mAP={self.best_map:.4f})")
        logger.info(f"   Will train epochs {self.start_epoch+1} → {self.cfg.EPOCHS}")
        return self.start_epoch

    def train_one_epoch(self, ep):
        self.model.train(); rl=0.0; nb=0; nan_ct=0; self.opt.zero_grad()
        pbar=tqdm(self.tl, desc=f"Ep {ep+1}/{self.cfg.EPOCHS} [TR]")
        for bi,(imgs,labs,_,_) in enumerate(pbar):
            imgs=imgs.to(self.dev,non_blocking=True)
            labs=labs.to(self.dev,non_blocking=True)
            if self.cfg.USE_MIXUP: imgs,labs=mixup_data(imgs,labs,self.cfg.MIXUP_ALPHA)
            if self.cfg.USE_AMP:
                with autocast():
                    if self.cfg.USE_CONTRASTIVE:
                        lcon,lcls=self.model(imgs,return_contrastive=True)
                        loss=self.crit(lcls,labs,lcon)
                    else: lcls=self.model(imgs); loss=self.crit(lcls,labs)
                    loss=loss/self.cfg.GRAD_ACCUM_STEPS
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_ct+=1; self.opt.zero_grad(); continue
                self.scaler.scale(loss).backward()
                if (bi+1)%self.cfg.GRAD_ACCUM_STEPS==0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                    ok=all(not(torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                           for p in self.model.parameters() if p.grad is not None)
                    if ok: self.scaler.step(self.opt)
                    else: nan_ct+=1
                    self.scaler.update(); self.opt.zero_grad()
                    if self.sched: self.sched.step()
            else:
                if self.cfg.USE_CONTRASTIVE:
                    lcon,lcls=self.model(imgs,return_contrastive=True)
                    loss=self.crit(lcls,labs,lcon)
                else: lcls=self.model(imgs); loss=self.crit(lcls,labs)
                loss=loss/self.cfg.GRAD_ACCUM_STEPS
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_ct+=1; self.opt.zero_grad(); continue
                loss.backward()
                if (bi+1)%self.cfg.GRAD_ACCUM_STEPS==0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                    self.opt.step(); self.opt.zero_grad()
                    if self.sched: self.sched.step()
            if self.ema: self.ema.update(self.model)
            rl+=loss.item()*self.cfg.GRAD_ACCUM_STEPS; nb+=1
            pbar.set_postfix(loss=f"{rl/nb:.4f}",nan=nan_ct)
        if nan_ct: logger.warning(f"{nan_ct} NaN batches")
        return rl/max(nb,1)

    @torch.no_grad()
    def validate(self, ep, use_ema=True):
        if self.ema and use_ema: self.ema.apply_shadow(self.model)
        self.model.eval(); rl=0.0; nb=0; aP=[]; aL=[]
        pbar=tqdm(self.vl,desc=f"Ep {ep+1}/{self.cfg.EPOCHS} [VL]")
        for imgs,labs,_,_ in pbar:
            imgs=imgs.to(self.dev,non_blocking=True); labs=labs.to(self.dev,non_blocking=True)
            lcls=self.model(imgs); loss=self.crit(lcls,labs)
            rl+=loss.item(); nb+=1
            p=torch.sigmoid(lcls).cpu().numpy()
            aP.append(np.nan_to_num(p,nan=0.5)); aL.append(labs.cpu().numpy())
        if self.ema and use_ema: self.ema.restore(self.model)
        aP=np.concatenate(aP); aL=np.concatenate(aL); preds=(aP>=0.5).astype(int)
        do_save=(ep==self.cfg.EPOCHS-1 or ep%max(1,self.cfg.EPOCHS//3)==0)
        metrics=compute_metrics(aL,aP,preds,Config.ALL_LABELS,
            save_dir=Config.CURVES_DIR if do_save else None, prefix=f"val_ep{ep+1}")
        return rl/max(nb,1), metrics["overall"]["AP"], aL, aP

    def save_ckpt(self, ep, mAP, name="best_model.pth"):
        path=os.path.join(self.cfg.CHECKPOINT_DIR,name)
        state={"epoch":ep,"model_state_dict":self.model.state_dict(),
               "optimizer_state_dict":self.opt.state_dict(),
               "val_mAP":mAP,"history":self.hist,"seed":self.cfg.SEED}
        if self.scaler: state["scaler_state_dict"]=self.scaler.state_dict()
        if self.ema: state["ema_shadow"]=self.ema.shadow
        torch.save(state,path); logger.info(f"Saved {name} (mAP={mAP:.4f})")

    def fit(self):
        logger.info(f"Training epochs {self.start_epoch+1} → {self.cfg.EPOCHS}")
        best_L,best_P=None,None
        for ep in range(self.start_epoch, self.cfg.EPOCHS):
            tl=self.train_one_epoch(ep)
            vl,mAP,vL,vP=self.validate(ep)
            lr=self.opt.param_groups[0]["lr"]
            self.hist["tl"].append(tl); self.hist["vl"].append(vl)
            self.hist["map"].append(mAP); self.hist["lr"].append(lr)
            logger.info(f"Ep {ep+1} — TR:{tl:.4f} VL:{vl:.4f} mAP:{mAP:.4f} LR:{lr:.2e}")
            if mAP>self.best_map:
                self.best_map=mAP; self.pat=0
                self.save_ckpt(ep,mAP,"best_model.pth"); best_L,best_P=vL,vP
            else: self.pat+=1
            self.save_ckpt(ep,mAP,"latest_model.pth")
            if self.pat>=self.cfg.EARLY_STOP_PATIENCE:
                logger.info(f"Early stop at ep {ep+1}"); break
        fig,ax=plt.subplots(1,3,figsize=(18,5))
        ax[0].plot(self.hist["tl"],label="Train"); ax[0].plot(self.hist["vl"],label="Val")
        ax[0].set_title("Loss"); ax[0].legend()
        ax[1].plot(self.hist["map"],'g-'); ax[1].set_title("Val mAP")
        ax[2].plot(self.hist["lr"],'orange'); ax[2].set_title("LR")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.LOGS_DIR,"curves.png"),dpi=150); plt.close()
        return self.best_map, best_L, best_P

# ============================================================================
# INFERENCE
# ============================================================================
@torch.no_grad()
def run_inference(model, vid_id, frame_dir, transform, device, bs=64):
    model.eval()
    ds=GalarSequentialVideoDataset(vid_id,frame_dir,transform)
    dl=DataLoader(ds,batch_size=bs,shuffle=False,
                  num_workers=4 if Config.MODE!="smoke" else 0,pin_memory=True)
    fns=[]; pbs=[]
    for imgs,fnums in tqdm(dl,desc=f"Infer {vid_id}"):
        imgs=imgs.to(device,non_blocking=True)
        p=torch.sigmoid(model(imgs)).cpu().numpy()
        pbs.append(np.nan_to_num(p,nan=0.5)); fns.extend(fnums.numpy().tolist())
    return np.array(fns), np.concatenate(pbs)

def process_predictions(fns, probs, thresholds=None, smooth_w=7, min_len=3):
    if thresholds is None: thresholds=np.full(Config.NUM_CLASSES,Config.CONFIDENCE_THRESH)
    bp=np.zeros_like(probs,dtype=int)
    for c in range(probs.shape[1]): bp[:,c]=(probs[:,c]>=thresholds[c]).astype(int)
    for c in range(bp.shape[1]):
        bp[:,c]=(median_filter(bp[:,c].astype(float),size=smooth_w)>0.5).astype(int)
    # Remove short events
    for c in range(bp.shape[1]):
        in_ev=False; st=0
        for i in range(len(bp)):
            if bp[i,c]==1 and not in_ev: in_ev=True; st=i
            elif bp[i,c]==0 and in_ev:
                in_ev=False
                if (i-st)<min_len: bp[st:i,c]=0
        if in_ev and (len(bp)-st)<min_len: bp[st:,c]=0
    # Merge close gaps
    mg=Config.MERGE_GAP
    for c in range(bp.shape[1]):
        in_ev=False; last_end=0
        for i in range(len(bp)):
            if bp[i,c]==1:
                if not in_ev and i-last_end<=mg and last_end>0: bp[last_end:i,c]=1
                in_ev=True
            else:
                if in_ev: last_end=i; in_ev=False
    return frame_preds_to_events_grouped(fns,bp,Config.ALL_LABELS), bp

# ============================================================================
# MAIN PIPELINES
# ============================================================================
def run_training_pipeline():
    all_videos=discover_all_videos()
    vids=sorted(all_videos.keys())
    if Config.MODE=="smoke":
        vids=vids[:Config.SMOKE_NUM_VIDEOS]; logger.info(f"SMOKE: {vids}")
    random.shuffle(vids)
    nv=max(1,int(len(vids)*Config.VAL_SPLIT))
    val_ids=vids[:nv]; train_ids=vids[nv:]
    logger.info(f"Train:{len(train_ids)} Val:{len(val_ids)}")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DifferentialBiomedCLIP(Config.MODEL_NAME,Config.NUM_CLASSES,device,
        Config.LAMBDA_INIT,Config.EB_REDUCTION,Config.ALL_LABELS,Config.USE_CONTRASTIVE)

    train_transform=build_train_transform(model.preprocess); val_transform=model.preprocess
    mf=Config.SMOKE_MAX_FRAMES if Config.MODE=="smoke" else None
    train_ds=GalarVideoDataset(train_ids,all_videos,train_transform,Config.FRAME_SAMPLE_RATE,mf)
    val_ds=GalarVideoDataset(val_ids,all_videos,val_transform,Config.VAL_SAMPLE_RATE,mf)
    if len(train_ds)==0: logger.error("Empty train!"); return
    pw=compute_class_weights(train_ds)

    kw=dict(pin_memory=True,drop_last=True,
            prefetch_factor=2 if Config.NUM_WORKERS>0 else None,
            persistent_workers=Config.NUM_WORKERS>0)
    if Config.USE_CLASS_BALANCED_SAMPLING:
        sampler=build_class_balanced_sampler(train_ds)
        tl=DataLoader(train_ds,Config.BATCH_SIZE,sampler=sampler,num_workers=Config.NUM_WORKERS,**kw)
    else:
        tl=DataLoader(train_ds,Config.BATCH_SIZE,shuffle=True,num_workers=Config.NUM_WORKERS,**kw)
    vl=DataLoader(val_ds,Config.BATCH_SIZE,shuffle=False,num_workers=Config.NUM_WORKERS,
                  pin_memory=True, prefetch_factor=2 if Config.NUM_WORKERS>0 else None,
                  persistent_workers=Config.NUM_WORKERS>0)

    cls_loss=AsymmetricFocalLoss(Config.FOCAL_GAMMA_POS,Config.FOCAL_GAMMA_NEG,0.05,pw,Config.LABEL_SMOOTHING)
    crit=CombinedLoss(cls_loss,Config.CONTRASTIVE_WEIGHT if Config.USE_CONTRASTIVE else 0,Config.CLS_WEIGHT,pw)

    bb_p,hd_p=[],[]
    for name,p in model.named_parameters():
        if any(k in name for k in ["classification_head","eb_block","dropout","lambda_","logit_scale"]):
            hd_p.append(p)
        else: bb_p.append(p)

    opt=torch.optim.AdamW([
        {"params":bb_p,"lr":Config.LR*Config.BACKBONE_LR_MULT},
        {"params":hd_p,"lr":Config.LR},
    ], weight_decay=Config.WEIGHT_DECAY)

    total_steps=Config.EPOCHS*len(tl)
    sched=OneCycleLR(opt,max_lr=[Config.LR*Config.BACKBONE_LR_MULT,Config.LR],
                     total_steps=total_steps,pct_start=0.15,
                     anneal_strategy='cos',div_factor=10,final_div_factor=100)

    trainer=Trainer(model,tl,vl,crit,opt,sched,device,Config)

    # ── RESUME ──
    if Config.RESUME_FROM and os.path.exists(Config.RESUME_FROM):
        logger.info(f"📂 Resuming from: {Config.RESUME_FROM}")
        trainer.resume_from_checkpoint(Config.RESUME_FROM)
    elif Config.RESUME_FROM:
        logger.warning(f"Resume path not found: {Config.RESUME_FROM} — starting fresh")

    best_mAP,best_L,best_P=trainer.fit()
    logger.info(f"Best val mAP: {best_mAP:.4f}")

    # Threshold optimization
    if best_L is not None:
        logger.info("Optimizing thresholds...")
        opt_t=optimize_thresholds(best_L,best_P,Config.ALL_LABELS)
        np.save(os.path.join(Config.CHECKPOINT_DIR,"optimal_thresholds.npy"),opt_t)
        logger.info("Final metrics with optimized thresholds:")
        opt_preds=np.zeros_like(best_P,dtype=int)
        for c in range(Config.NUM_CLASSES): opt_preds[:,c]=(best_P[:,c]>=opt_t[c]).astype(int)
        compute_metrics(best_L,best_P,opt_preds,Config.ALL_LABELS,Config.CURVES_DIR,"val_final_opt")

    # Val video events
    logger.info("Generating validation events...")
    ckpt=torch.load(os.path.join(Config.CHECKPOINT_DIR,"best_model.pth"),map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "ema_shadow" in ckpt:
        for n,p in model.named_parameters():
            if n in ckpt["ema_shadow"]: p.data.copy_(ckpt["ema_shadow"][n])
        logger.info("Applied EMA weights")
    thresh=np.load(os.path.join(Config.CHECKPOINT_DIR,"optimal_thresholds.npy")) if best_L is not None else np.full(Config.NUM_CLASSES,0.5)
    all_ev={}
    for vid in val_ids:
        fns,probs=run_inference(model,vid,all_videos[vid],val_transform,device,Config.BATCH_SIZE*2)
        evts,_=process_predictions(fns,probs,thresh,Config.SMOOTH_WINDOW,Config.MIN_EVENT_LENGTH)
        all_ev[vid]=evts; logger.info(f"  Video {vid}: {len(evts)} events")
    build_submission_json(all_ev,os.path.join(Config.RESULTS_DIR,"val_predictions.json"))
    logger.info("="*60+"\nTRAINING COMPLETE\n"+"="*60)

def run_test_pipeline():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DifferentialBiomedCLIP(Config.MODEL_NAME,Config.NUM_CLASSES,device,
        Config.LAMBDA_INIT,Config.EB_REDUCTION,Config.ALL_LABELS,Config.USE_CONTRASTIVE)
    transform=model.preprocess
    ckpt_path=os.path.join(Config.CHECKPOINT_DIR,"best_model.pth")
    if not os.path.exists(ckpt_path): logger.error(f"No ckpt at {ckpt_path}"); return
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "ema_shadow" in ckpt:
        for n,p in model.named_parameters():
            if n in ckpt["ema_shadow"]: p.data.copy_(ckpt["ema_shadow"][n])
        logger.info("Applied EMA weights")
    logger.info(f"Loaded ep {ckpt['epoch']+1} (mAP={ckpt['val_mAP']:.4f})")

    tp=os.path.join(Config.CHECKPOINT_DIR,"optimal_thresholds.npy")
    thresh=np.load(tp) if os.path.exists(tp) else np.full(Config.NUM_CLASSES,Config.CONFIDENCE_THRESH)
    logger.info(f"Thresholds: {['%.2f'%t for t in thresh]}")

    test_vids=discover_test_videos()
    if not test_vids: logger.error("No test videos"); return
    all_ev={}; timing={}
    for vid,fdir in test_vids.items():
        t0=time.time()
        fns,probs=run_inference(model,vid,fdir,transform,device,Config.BATCH_SIZE*2)
        evts,bp=process_predictions(fns,probs,thresh,Config.SMOOTH_WINDOW,Config.MIN_EVENT_LENGTH)
        elapsed=time.time()-t0; timing[vid]=elapsed; all_ev[vid]=evts
        logger.info(f"  {vid}: {len(evts)} events, {len(fns)} frames, {elapsed:.1f}s")
        csv_path=os.path.join(Config.RESULTS_DIR,f"test_{vid}_frames.csv")
        df=pd.DataFrame(bp,columns=Config.ALL_LABELS); df.insert(0,"index",fns)
        df.to_csv(csv_path,index=False)
        lc=defaultdict(int)
        for e in evts:
            for l in e["label"]: lc[l]+=1
        logger.info(f"    Labels: {dict(lc)}")

    sub=build_submission_json(all_ev,os.path.join(Config.RESULTS_DIR,"test_predictions.json"))
    rows=[]
    for v in sub["videos"]:
        for e in v["events"]:
            rows.append({"video_id":v["video_id"],"start":e["start"],
                         "end":e["end"],"label":str(e["label"])})
    pd.DataFrame(rows).to_excel(os.path.join(Config.RESULTS_DIR,"test_predictions.xlsx"),index=False)
    tot=sum(timing.values())
    logger.info(f"\n{'='*60}\nTEST COMPLETE")
    for v,t in timing.items(): logger.info(f"  {v}: {t:.1f}s")
    logger.info(f"  Total: {tot:.1f}s ({tot/60:.1f}min)\n{'='*60}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    seed_everything(Config.SEED); Config.apply_mode()
    logger.info("="*60)
    logger.info(f"ICPR 2026 RARE-VISION — Diff-BiomedCLIP v3.1")
    logger.info(f"Mode:{Config.MODE} Seed:{Config.SEED}")
    if torch.cuda.is_available():
        logger.info(f"GPU:{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB)")
    logger.info(f"Ep:{Config.EPOCHS} BS:{Config.BATCH_SIZE} LR:{Config.LR} WD:{Config.WEIGHT_DECAY}")
    logger.info(f"AMP:{Config.USE_AMP} Mixup:{Config.USE_MIXUP}(α={Config.MIXUP_ALPHA}) "
                f"EMA:{Config.USE_EMA}({Config.EMA_DECAY}) LS:{Config.LABEL_SMOOTHING}")
    logger.info(f"Aug:{Config.USE_STRONG_AUG} Balanced:{Config.USE_CLASS_BALANCED_SAMPLING}")
    logger.info(f"BackboneLR:{Config.LR*Config.BACKBONE_LR_MULT:.2e} Dropout:{Config.DROPOUT}")
    if Config.RESUME_FROM: logger.info(f"RESUME: {Config.RESUME_FROM}")
    logger.info(f"Output:{Config.OUTPUT_DIR}")
    logger.info("="*60)
    if Config.MODE in ("smoke","train"): run_training_pipeline()
    elif Config.MODE == "test": run_test_pipeline()

if __name__ == "__main__": main()
else: main()
