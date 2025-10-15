import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ConvBNReLU(nn.Module):
    """
    Basic Conv → BN → ReLU block.
    Inputs:
      x: [B, Cin, H, W]
    Returns:
      [B, Cout, H, W]
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpFuse(nn.Module):
    """
    Fuse a low-resolution deep feature with a higher-resolution skip feature.
    Inputs:
      low_res: deep, low-res feature map
        [B, C1, H1, W1]   (e.g., stride-32)
      skip:    shallower, higher-res feature map
        [B, C2, H2, W2]   (e.g., stride-16)
    Output:
      [B, Co, H2, W2]
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch)
        )
    def forward(self, low_res, skip):
        low_res = F.interpolate(low_res, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, low_res], dim=1)
        x = self.refine(x)
        return x


class ImageEncoder(nn.Module):
    """
    EfficientNet-B0 backbone → fused features → depth and context maps.
    Inputs:
      x: [B, 3, H, W]
    Returns:
      depth_prob: [B, D, H/16, W/16]
      lifted:     [B, C, D, H/16, W/16]
    """
    def __init__(self, depth_bins, context_dim=64   ):
        super().__init__()
        
        self.D = depth_bins
        self.C = context_dim

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features

        # Fuse EfficientNet’s 1/32 (320ch) and 1/16 (112ch) features, compress 432→256 at 1/16
        self.fuse = UpFuse(in_ch=112+320, out_ch=512)

        # 1x1 Conv heads
        self.depth_head = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_effnet_feats(self, x):
        """
        Extract stride-16 and stride-32 features.
        Inputs:
          x: [B, 3, H, W]
        Returns:
          red4: [B, 112, H/16, W/16]
          red5: [B, 320, H/32, W/32]
        """
        red4, red5 = None, None
        h0, w0 = x.shape[-2:]
        cur_h, cur_w = h0, w0
        cur_stride = 1

        for layer in self.backbone:
            x = layer(x)
            h, w = x.shape[-2:]
            if (h, w) != (cur_h, cur_w):
                ds = (cur_h // h)
                cur_stride *= ds
                cur_h, cur_w = h, w

            C = x.shape[1]
            if cur_stride == 16 and C == 112: red4 = x
            if cur_stride == 32 and C == 320: red5 = x
        return red4, red5
        
    def get_fused_feat(self, x):
        """
        Fuse stride-32 into stride-16 features.
        Inputs:
          x: [B, 3, H, W]
        Returns:
          [B, 512, H/16, W/16]
        """
        red4, red5 = self.get_effnet_feats(x)
        return self.fuse(low_res=red5, skip=red4) # -> [B,512,H/16,W/16]
    
    def forward(self, x):
        fused_feats = self.get_fused_feat(x)             # [B,512,h,w]
        dc = self.depth_head(fused_feats)                # [B, D+C, h, w]
        depth_logits = dc[:, :self.D, ...]               # [B, D, h, w]
        ctx_logits = dc[:, self.D:self.D+self.C, ...]    # [B, C, h, w]
        depth_prob = depth_logits.softmax(dim=1)         # α over D
        lifted = depth_prob.unsqueeze(1) * ctx_logits.unsqueeze(2) # [B, C, D, h, w]
        return depth_prob, lifted

class Lift(nn.Module):
    """
    Lift stage: frustum grid + camera encoding.
    Inputs:
      depth_bins, context_dim, dbound, final_dims, downsample
    Returns:
      create_frustum() -> [D, h, w, 3]
      forward(...)     -> depth_prob, cam_feats, pts_ego
    """
    def __init__(self, depth_bins, context_dim=64, 
                dbound=(4.0, 45.0, 1.0), final_dims=(256, 704), 
                downsample=16):
        super().__init__()
        self.D = depth_bins
        self.C = context_dim
        self.dbound = dbound  # (d_min, d_max, d_step)
        self.final_dims = final_dims
        self.downsample = downsample
        self.encoder = ImageEncoder(self.D, self.C)
        frustum = self.create_frustum()   # [D,h,w,3]
        self.register_buffer("frustum", frustum, persistent=True)

    def create_frustum(self):
        """
        Build (u,v,d) grid at encoder scale.
        Returns:
          [D, h, w, 3]
        """
        H, W = self.final_dims
        h, w = H // self.downsample, W // self.downsample
        dmin, dmax, dstep = self.dbound
        ds = torch.arange(dmin, dmax+1e-5, dstep, dtype=torch.float32)
        us = torch.linspace(0, W-1, w, dtype=torch.float32)
        vs = torch.linspace(0, H-1, h, dtype=torch.float32)
        uu = us.view(1, 1, w).expand(self.D, h, w)
        vv = vs.view(1, h, 1).expand(self.D, h, w)
        dd = ds.view(self.D, 1, 1).expand(self.D, h, w)
        return torch.stack([uu, vv, dd], dim=-1)  # [D, h, w, 3]


    @torch.no_grad()
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        Map (u,v,d) samples to ego coords.
        Inputs:
          rots:      [B, N, 3, 3]  (cam->ego rotation)
          trans:     [B, N, 3]
          intrins:   [B, N, 3, 3]
          post_rots: [B, N, 2, 2]  (image post-aug rot/scale)
          post_trans:[B, N, 2]     (image post-aug translation)
        Returns:
          pts_ego:   [B, N, D, h, w, 3]
        """
        device = rots.device
        fr = self.frustum.to(device)
        B, N = trans.shape[:2]
        D, h, w, _ = fr.shape

        uv = fr[..., :2][None, None]     # [1,1,D,h,w,2]
        dd = fr[..., 2:3][None, None]    # [1,1,D,h,w,1]

        # Undo image post-augmentation
        uv = uv - post_trans.view(B, N, 1, 1, 1, 2)
        inv_post = torch.linalg.inv(post_rots).view(B, N, 1, 1, 1, 2, 2)
        uv = (inv_post @ uv.unsqueeze(-1)).squeeze(-1)  # [B,N,D,h,w,2]

        # Back-project with intrinsics (pinhole)
        pix = torch.cat([uv, torch.ones_like(dd)], dim=-1) * dd  # [B,N,D,h,w,3]
        invK = torch.linalg.inv(intrins).view(B, N, 1, 1, 1, 3, 3)

        # Camera -> ego
        combine = (rots @ invK).view(B, N, 1, 1, 1, 3, 3)
        pts_ego = (combine @ pix.unsqueeze(-1)).squeeze(-1) + trans.view(B, N, 1, 1, 1, 3)
        return pts_ego

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans):
        """
        Encode images and compute geometry.
        Inputs:
          imgs: [B,N,3,H,W], rots, trans, intrins, post_rots, post_trans
        Returns:
          depth_prob: [B,N,D,h,w]
          cam_feats:  [B,N,D,h,w,C]
          pts_ego:    [B,N,D,h,w,3]
        """
        B, N, _, H, W = imgs.shape

        x = imgs.flatten(0, 1)
        depth_prob, lifted = self.encoder(x)  # [B*N,D,h,w], [B*N,C,D,h,w]

        BN, D, h, w = depth_prob.shape
        depth_prob = depth_prob.reshape(B, N, D, h, w)
        cam_feats  = lifted.permute(0, 2, 3, 4, 1).reshape(B, N, D, h, w, self.C)

        pts_ego = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        return depth_prob, cam_feats, pts_ego
