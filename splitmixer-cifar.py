import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.modules.container import ModuleList

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()

""" Dataset args """
ds = "cifar10"  # Choose from ['cifar10', 'cifar100'] # @param {isTemplate:true}
args_batch_size = 512
args_scale = 1
args_reprob = 0
args_ra_m = 12
args_ra_n = 2
args_jitter = 0
res = 32
num_classes = 10 if ds == "cifar10" else 100

""" Model args """
args_name = "SplitMixer"  # Choose from 'SplitMixer', 'StrideMixer', 'ConvMixer' # @param {isTemplate:true}
args_mixer_setting = (
    "I"  # need to provide this UNLESS using ConvMixer # @param {isTemplate:true}
)
args_ratio = 2 / 3  # need to provide this if using I, V # @param {isTemplate:true}
args_segments = (
    2  # need to provide this if using II, III, IV # @param {isTemplate:true}
)
args_spatial_trick = True
args_channel_trick = True

args_hdim = 256
args_blocks = 8
args_psize = 2
args_conv_ks = 5

rt_sg = (
    f"-{args_ratio:.2f}" if args_mixer_setting in ["I", "V"] else f"-{args_segments}"
)
if args_name == "ConvMixer":
    args_mixer_setting = ""
    args_spatial_trick = False
    args_channel_trick = False
    rt_sg = ""
sp = "-sp" if args_spatial_trick else ""  # indicating using spatial trick
ch = "-ch" if args_channel_trick else ""  # indicating using channel trick
model_sig = f"{args_name}{args_mixer_setting}{rt_sg}{sp}{ch}"
print(model_sig)

""" Training args"""
args_lr_max = 0.05
args_wd = 0.005
args_epochs = 100
args_clip_norm = True
args_workers = 2


# --------------------------- Building Blocks ----------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ChannelMixerI(nn.Module):
    """Partial overlap; In each block only one segment is convolved."""

    def __init__(self, hdim, is_odd=0, ratio=2 / 3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.partial_c = int(hdim * ratio)
        self.mixer = nn.Conv2d(self.partial_c, self.partial_c, kernel_size=1)
        self.is_odd = is_odd

    def forward(self, x):
        if self.is_odd == 0:
            idx = self.partial_c
            return torch.cat((self.mixer(x[:, :idx]), x[:, idx:]), dim=1)
        else:
            idx = self.hdim - self.partial_c
            return torch.cat((x[:, :idx], self.mixer(x[:, idx:])), dim=1)


class ChannelMixerII(nn.Module):
    """No overlap; In each block only one segment is convolved."""

    def __init__(self, hdim, remainder=0, num_segments=3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.remainder = remainder
        self.num_segments = num_segments
        self.bin_dim = int(hdim / num_segments)
        self.c = (
            hdim - self.bin_dim * (num_segments - 1)
            if (remainder == num_segments - 1)
            else self.bin_dim
        )
        self.mixer = nn.Conv2d(self.c, self.c, kernel_size=1)

    def forward(self, x):
        start = self.remainder * self.bin_dim
        end = (
            self.hdim
            if (self.remainder == self.num_segments - 1)
            else ((self.remainder + 1) * self.bin_dim)
        )
        return torch.cat((x[:, :start], self.mixer(x[:, start:end]), x[:, end:]), dim=1)


class ChannelMixerIII(nn.Module):
    """No overlap; In each block all segments are convolved;
    Parameters are shared across segments."""

    def __init__(self, hdim, num_segments=3, **kwargs):
        super().__init__()
        assert (
            hdim % num_segments == 0
        ), f"hdim {hdim} need to be divisible by num_segments {num_segments}"
        self.hdim = hdim
        self.num_segments = num_segments
        self.c = hdim // num_segments
        self.mixer = nn.Conv2d(self.c, self.c, kernel_size=1)

    def forward(self, x):
        c = self.c
        x = [self.mixer(x[:, c * i : c * (i + 1)]) for i in range(self.num_segments)]
        return torch.cat(x, dim=1)


class ChannelMixerIV(nn.Module):
    """No overlap; In each block all segments are convolved;
    No parameter sharing across segments."""

    def __init__(self, hdim, num_segments=3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.num_segments = num_segments
        c = hdim // num_segments
        last_c = hdim - c * (num_segments - 1)
        self.mixer = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=1) for _ in range(num_segments - 1)]
            + ([nn.Conv2d(last_c, last_c, kernel_size=1)])
        )
        self.c, self.last_c = c, last_c

    def forward(self, x):
        c, last_c = self.c, self.last_c
        x = [
            self.mixer[i](x[:, c * i : c * (i + 1)])
            for i in (range(self.num_segments - 1))
        ] + [self.mixer[-1](x[:, -last_c:])]
        return torch.cat(x, dim=1)


class ChannelMixerV(nn.Module):
    """Partial overlap; In each block all segments are convolved;
    No parameter sharing across segments."""

    def __init__(self, hdim, ratio=2 / 3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.c = int(hdim * ratio)
        self.mixer1 = nn.Conv2d(self.c, self.c, kernel_size=1)
        self.mixer2 = nn.Conv2d(self.c, self.c, kernel_size=1)

    def forward(self, x):
        c, hdim = self.c, self.hdim
        x = torch.cat((self.mixer1(x[:, :c]), x[:, c:]), dim=1)
        return torch.cat((x[:, : (hdim - c)], self.mixer2(x[:, (hdim - c) :])), dim=1)


# ------------------------------ Main Model ------------------------------------


class SplitMixer(nn.Module):
    def __init__(
        self,
        hdim,
        num_blocks,
        kernel_size=5,
        patch_size=2,
        num_classes=10,
        ratio=2 / 3,
        num_segments=2,
        img_size=32,
        mixer_setting="I",
        spatial_trick=True,
        channel_trick=True,
        act_layer=nn.GELU,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = hdim

        # n_patch = img_size // patch_size
        self.patch_emb = nn.Sequential(
            nn.Conv2d(3, hdim, kernel_size=patch_size, stride=patch_size),
            act_layer(),
            nn.BatchNorm2d(hdim),
            # nn.LayerNorm([hdim, n_patch, n_patch]),
        )

        self.mixer_blocks = ModuleList()

        for i in range(num_blocks):
            spatial_mixer = (
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            hdim, hdim, (1, kernel_size), groups=hdim, padding="same"
                        ),
                        act_layer(),
                        nn.BatchNorm2d(hdim),
                        # nn.LayerNorm([hdim, n_patch, n_patch]),
                        nn.Conv2d(
                            hdim, hdim, (kernel_size, 1), groups=hdim, padding="same"
                        ),
                        act_layer(),
                        nn.BatchNorm2d(hdim),
                        # nn.LayerNorm([hdim, n_patch, n_patch]),
                    )
                )
                if spatial_trick
                else Residual(
                    nn.Sequential(
                        nn.Conv2d(hdim, hdim, kernel_size, groups=hdim, padding="same"),
                        act_layer(),
                        nn.BatchNorm2d(hdim),
                    )
                )
            )

            self.mixer_blocks.append(spatial_mixer)

            if channel_trick:
                mixer_args = {
                    "hdim": hdim,
                    "is_odd": i % 2,
                    "ratio": ratio,
                    "remainder": i % num_segments,
                    "num_segments": num_segments,
                }
                channel_mixer = nn.Sequential(
                    globals()[f"ChannelMixer{mixer_setting}"](**mixer_args),
                    act_layer(),
                    nn.BatchNorm2d(hdim),
                    # nn.LayerNorm([hdim, n_patch, n_patch]),
                )
            else:
                channel_mixer = nn.Sequential(
                    nn.Conv2d(hdim, hdim, kernel_size=1),
                    act_layer(),
                    nn.BatchNorm2d(hdim),
                )
            self.mixer_blocks.append(channel_mixer)

        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.head = nn.Linear(hdim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_emb(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.pooling(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class StrideMixer(nn.Module):
    def __init__(
        self,
        hdim,
        num_blocks,
        kernel_size=5,
        patch_size=2,
        num_classes=10,
        channel_kernel_size=128,
        channel_stride=128,
        **kwargs,
    ):
        super().__init__()
        self.patch_emb = nn.Sequential(
            nn.Conv2d(3, hdim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hdim),
        )

        # calculate channel conv h_out
        c_out = (hdim - channel_kernel_size) / channel_stride + 1
        assert (
            hdim % c_out == 0
        ), "setting is not valid, double check channel_kernel_size and channel_stride"
        h_out = int(hdim / c_out)

        self.spatial_mixer, self.channel_mixer = ModuleList(), ModuleList()
        for _ in range(num_blocks):
            spatial_mixer = Residual(
                nn.Sequential(
                    nn.Conv2d(
                        hdim, hdim, (1, kernel_size), groups=hdim, padding="same"
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(hdim),
                    nn.Conv2d(
                        hdim, hdim, (kernel_size, 1), groups=hdim, padding="same"
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(hdim),
                )
            )
            self.spatial_mixer.append(spatial_mixer)

            channel_mixer = nn.Sequential(
                nn.Conv3d(
                    1, h_out, (channel_kernel_size, 1, 1), stride=(channel_stride, 1, 1)
                ),
                nn.GELU(),
                nn.Flatten(1, 2),
                nn.BatchNorm2d(hdim),
            )
            self.channel_mixer.append(channel_mixer)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(hdim, num_classes)
        )

        self.num_blocks = num_blocks

    def forward(self, x):
        x = self.patch_emb(x)
        for i in range(self.num_blocks):
            x = self.spatial_mixer[i](x).unsqueeze(1)
            x = self.channel_mixer[i](x)
        return self.head(x)


mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
}

std = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761),
}

mean, std = mean[ds], std[ds]

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(args_scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args_ra_n, magnitude=args_ra_m),
        transforms.ColorJitter(args_jitter, args_jitter, args_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=args_reprob),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)


trainset = getattr(torchvision.datasets, ds.upper())(
    root="./data", train=True, download=True, transform=train_transform
)
testset = getattr(torchvision.datasets, ds.upper())(
    root="./data", train=False, download=True, transform=test_transform
)
print(
    "\nNumber of images in the train set:",
    len(trainset),
    "; Number of images in the test set:",
    len(testset),
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args_batch_size, shuffle=True, num_workers=args_workers
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args_batch_size, shuffle=False, num_workers=args_workers
)


""" Create and print the model based on args. """

model_kwargs = {
    "kernel_size": args_conv_ks,
    "patch_size": args_psize,
    "num_classes": num_classes,
    "ratio": args_ratio,
    "num_segments": args_segments,
    "mixer_setting": args_mixer_setting,
    "spatial_trick": args_spatial_trick,
    "channel_trick": args_channel_trick,
}

if args_name == "StrideMixer":
    if args_mixer_setting == "III":
        seg_len = args_hdim // args_segments  # must be divisible for setting III!!!
        model_kwargs["channel_kernel_size"] = seg_len
        model_kwargs["channel_stride"] = seg_len

    else:  # SettingI's extension (a ghost setting VI...)
        seg_len = int(args_hdim * args_ratio)
        model_kwargs["channel_kernel_size"] = seg_len
        model_kwargs["channel_stride"] = args_hdim - seg_len

if args_name == "ConvMixer":
    args_name = "SplitMixer"

model = globals()[args_name](args_hdim, args_blocks, **model_kwargs).to(device)
print(model)

if device.type == "cuda":
    model = nn.DataParallel(model).cuda()

lr_schedule = lambda t: np.interp(
    [t],
    [0, args_epochs * 2 // 5, args_epochs * 4 // 5, args_epochs],
    [0, args_lr_max, args_lr_max / 20.0, 0],
)[0]

opt = optim.AdamW(model.parameters(), lr=args_lr_max, weight_decay=args_wd)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(args_epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()

        lr = lr_schedule(epoch + (i + 1) / len(trainloader))
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        if args_clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    print(
        f"[{model_sig}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}"
    )
