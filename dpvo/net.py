import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

DIM = 384

DIM = 384

class Update_Failed(nn.Module):
    def __init__(self, p):
        super(Update_Failed, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        # Use original SoftAggONNX for DPVO, CV28-friendly version will be used in ONNX_COMPATIBLE=True
        # self.agg_kk = SoftAggONNX_ori(DIM)
        # self.agg_ij = SoftAggONNX_ori(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())

        # CV28-friendly aggregation
        self.agg_kk = SoftAggCV28Friendly(DIM)
        self.agg_ij = SoftAggCV28Friendly(DIM)

    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        flow: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        kk: torch.Tensor,
        ONNX_COMPATIBLE: bool = True,
        kk_idx_map: torch.Tensor = None,
        G_kk: int = None,
        ij_idx_map: torch.Tensor = None,
        G_ij: int = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, None]]:

        # CV28 input reshape (4D -> 3D)
        net  = net.squeeze(-1).permute(0, 2, 1)    # [B, N, DIM]
        inp  = inp.squeeze(-1).permute(0, 2, 1)
        corr = corr.squeeze(-1).permute(0, 2, 1)

        ii = ii.squeeze(0).squeeze(-1)
        jj = jj.squeeze(0).squeeze(-1)
        kk = kk.squeeze(0).squeeze(-1)

        if not ONNX_COMPATIBLE:
            # Original DPVO branch
            net = net + inp + self.corr(corr)
            net = self.norm(net)

            ix, jx = neighbors_tensor(kk, jj)
        
            mask_ix = (ix >= 0).float().reshape(1, -1, 1)
            mask_jx = (jx >= 0).float().reshape(1, -1, 1)

            net = net + self.c1(mask_ix * net[:,ix])
            net = net + self.c2(mask_jx * net[:,jx])

            # Original code
            # net = net + self.agg_kk(net, kk)
            # net = net + self.agg_ij(net, ii*12345 + jj)
            # 🔥 CV28-friendly soft aggregation
            net = net + self.agg_kk(net, kk_idx_map, G_kk)
            net = net + self.agg_ij(net, ij_idx_map, G_ij)

            net = self.gru(net)
            
            return net, (self.d(net), self.w(net), None)
        else:
            # ONNX/CV28-compatible branch
            # Precomputed index maps and G must be provided
            assert kk_idx_map is not None and G_kk is not None, "kk_idx_map and G_kk required for CV28"
            assert ij_idx_map is not None and G_ij is not None, "ij_idx_map and G_ij required for CV28"

            # Minimal bias on ii to prevent ONNX optimization removal
            ii_float = ii.float()
            ii_bias = (ii_float.sum() * 1e-10).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            ii_bias = ii_bias.expand_as(net)
            net = net + inp + self.corr(corr) + ii_bias
            net = self.norm(net)

            # compute neighbors
            ix, jx = neighbors_tensor(kk, jj)
            H = net.shape[1]
            ix = torch.clamp(ix, 0, H-1)
            jx = torch.clamp(jx, 0, H-1)
            ix = torch.abs(ix)
            jx = torch.abs(jx)
            ix = torch.clamp(ix, 0, H-1)
            jx = torch.clamp(jx, 0, H-1)

            # neighbor updates
            net = net + self.c1(net[:, ix])
            net = net + self.c2(net[:, jx])

            # 🔥 CV28-friendly soft aggregation
            net = net + self.agg_kk(net, kk_idx_map, G_kk)
            net = net + self.agg_ij(net, ij_idx_map, G_ij)

            net = self.gru(net)

            # outputs
            d_out = self.d(net)
            w_out = self.w(net)

            # minimal ii contribution to outputs for ONNX dependency
            ii_final = ii.float().unsqueeze(0).unsqueeze(-1)
            ii_scale = 1e-10
            d_out = d_out + (ii_final.expand(1, -1, 2) * ii_scale)
            w_out = w_out + (ii_final.expand(1, -1, 2) * ii_scale)

            return net, (d_out, w_out, None)



class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAggONNX(DIM)
        self.agg_ij = SoftAggONNX(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())

    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        flow: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        kk: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, None]]:
        """
        Update operator for DPVO factor graph optimization.
        Notes:
            B is batch size (typically 1)
            N is the number of edges/patches
            DIM=384.
        Args:
            net: torch.Tensor - Network hidden state, shape [B, N, DIM]. Type: float32
            inp: torch.Tensor - Input features (imap), shape [B, N, DIM]. Type: float32
            corr: torch.Tensor - Correlation features, shape [B, N, corr_dim] where corr_dim = 2*49*p*p (882 for p=3). Type: float32
            flow: torch.Tensor - Optical flow (currently not used in the forward pass but kept for API compatibility), shape [B, N, 2]. Type: float32
            ii: torch.Tensor - Source frame indices, shape [N] or [B, N]. Type: int64
            jj: torch.Tensor - Target frame indices, shape [N] or [B, N]. Type: int64
            kk: torch.Tensor - Patch indices, shape [N] or [B, N]. Type: int64
        
        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, None]]:
                - net: Updated network state, shape [B, N, DIM]
                - (delta, weight, None): Tuple containing:
                    - delta: Position correction (predicted deltas), shape [B, N, 2]
                    - weight: Confidence weights, shape [B, N, 2]
                    - None: Placeholder (for compatibility with other code)
        """
        ONNX_COMPATIBLE = True
        # if not ONNX_COMPATIBLE:
            # net = net + inp + self.corr(corr)
            # net = self.norm(net)

            # # ix, jx = fastba.neighbors(kk, jj)
            # ix, jx = neighbors_tensor(kk, jj)
        
            # mask_ix = (ix >= 0).float().reshape(1, -1, 1)
            # mask_jx = (jx >= 0).float().reshape(1, -1, 1)

            # net = net + self.c1(mask_ix * net[:,ix])
            # net = net + self.c2(mask_jx * net[:,jx])

            # net = net + self.agg_kk(net, kk)
            # net = net + self.agg_ij(net, ii*12345 + jj)

            # net = self.gru(net)
            
            # return net, (self.d(net), self.w(net), None)
        if ONNX_COMPATIBLE:
            # CRITICAL: Ensure ii is used early to prevent ONNX from optimizing it away
            # Convert to float and create a dependency that affects the computation
            # ii_float = ii.float()  # Convert to float for operations
            # Create a minimal bias tensor from ii that gets added to net
            # Use a very small but non-zero value to create dependency chain
            # ii_bias = (ii_float.sum() * 1e-10).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # ii_bias = ii_bias.expand_as(net)  # [1, H, DIM]
            # Add to net - this creates a dependency that ONNX cannot optimize away
            net = net + inp + self.corr(corr) # + ii_bias
            net = self.norm(net)

            # compute neighbors
            ix, jx = neighbors_tensor(kk, jj)

            # mask_ix = torch.ones_like(ix, dtype=net.dtype).view(1, -1, 1)  # use full mask
            # mask_jx = torch.ones_like(jx, dtype=net.dtype).view(1, -1, 1)

            # clamp indices to [0, H-1] for CV28 to avoid negative indices
            # AMBA CV28 doesn't handle negative indices correctly in Gather operations
            H = net.shape[1]
            ix = torch.clamp(ix, min=0, max=H-1)
            jx = torch.clamp(jx, min=0, max=H-1)
            
            # Ensure indices are non-negative and valid (double-check for safety)
            ix = torch.abs(ix)  # Ensure non-negative
            jx = torch.abs(jx)  # Ensure non-negative
            ix = torch.clamp(ix, min=0, max=H-1)
            jx = torch.clamp(jx, min=0, max=H-1)

            # neighbor updates
            net = net + self.c1(net[:, ix])
            net = net + self.c2(net[:, jx])

            # soft aggregation
            # Ensure ii is used to prevent ONNX from optimizing it away
            # Create a dependency on ii that affects the computation
            # ii_used = ii.to(torch.int32)  # Ensure ii is processed
            # ii_used = torch.clamp(ii_used, min=0, max=net.shape[1]-1)  # Clamp to valid range
            
            # net = net + self.agg_kk(net, kk)
            # # Use ii_used instead of ii directly to create a dependency chain
            # net = net + self.agg_ij(net, ii_used*12345 + jj)
            
            # soft aggregation with valid indices
            net = net + self.agg_kk(net, kk)
            net = net + self.agg_ij(net, ii)  # <-- do NOT multiply or offset indices

            net = self.gru(net)

            # outputs
            d_out = self.d(net)
            w_out = self.w(net)
            
            # CRITICAL: Ensure ii affects outputs to prevent ONNX from optimizing it away
            # Add minimal contribution based on ii to create dependency chain
            # This ensures ONNX sees ii as required for the outputs
            ii_final = ii.float().unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
            ii_scale = 1e-10 # 1e-16  # Very small but non-zero to create dependency
            
            # Add tiny contribution to outputs based on ii
            # This creates a dependency that ONNX cannot optimize away
            d_out = d_out + (ii_final.expand(1, -1, 2) * ii_scale)
            w_out = w_out + (ii_final.expand(1, -1, 2) * ii_scale)

            return net, (d_out, w_out, None)
       
            


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        print(f"fmap.shape :{b},{n},{c},{h},{w}")
        P = self.patch_size

        # bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'RANDOM':
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing='ij')
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"), indexing='ij')
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"), indexing='ij')

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

        
"""
----------------------------------
For convet to ONNX function
-----------------------------------
"""

def neighbors_tensor(ii: torch.Tensor, jj: torch.Tensor):
    """
    Fully tensorized neighbors computation WITHOUT torch.unique (ONNX-compatible)
    ii, jj: [N]  (group indices)
    Returns:
        ix, jx: previous and next neighbor indices, shape [N], always in [0, N-1]
    """
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    N = ii.shape[0]

    # Broadcast ii and jj to compare all pairs
    ii_row = ii.unsqueeze(0).expand(N, N)
    ii_col = ii.unsqueeze(1).expand(N, N)
    jj_row = jj.unsqueeze(0).expand(N, N)
    jj_col = jj.unsqueeze(1).expand(N, N)

    # Mask to only allow neighbors in the same group
    mask = ii_row == ii_col  # [N, N]

    # Previous neighbor: jj_row < jj_col
    prev_mask = mask & (jj_row < jj_col)
    prev_jj = torch.where(prev_mask, jj_row, torch.full_like(jj_row, 0))
    ix = torch.argmax(prev_jj, dim=1)
    ix = torch.clamp(ix, min=0, max=N-1)  # ✅ clamp for CV28
    

    # Next neighbor: jj_row > jj_col
    next_mask = mask & (jj_row > jj_col)
    next_jj = torch.where(next_mask, jj_row, torch.full_like(jj_row, N))
    jx = torch.argmin(next_jj, dim=1)
    jx = torch.clamp(jx, min=0, max=N-1)  # ✅ clamp for CV28

    return ix.to(torch.int32), jx.to(torch.int32)


class SoftAggONNX_worng_result(nn.Module):
    def __init__(self, dim=512, expand=True):
        super().__init__()
        self.expand = expand
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x, ix):
        """
        AMBA CV28-compatible SoftAgg WITHOUT scatter operations.
        Simplified version that avoids ScatterElements, Unique, and ConstantOfShape.
        x  : [B, N, C]
        ix : [N]  (group indices, e.g., kk/ii/jj) - used for compatibility but not for actual grouping
        Returns: [B, N, C] if expand=True
        """
        B, N, C = x.shape
        
        # Simplified approach: Apply transformations without scatter grouping
        # This avoids all ScatterElements operations that AMBA CV28 doesn't support
        
        # Step 1: Compute logits and element-wise softmax
        logits = self.g(x)  # [B, N, C]
        # Use stable softmax computation
        logits_max = logits.max(dim=1, keepdim=True)[0]  # [B, 1, C]
        exp_logits = torch.exp(logits - logits_max)  # [B, N, C]
        exp_sum = exp_logits.sum(dim=1, keepdim=True)  # [B, 1, C]
        w = exp_logits / torch.clamp(exp_sum, min=1e-6)  # [B, N, C]
        
        # Step 2: Apply weighted transformation
        fx = self.f(x)  # [B, N, C]
        weighted = fx * w  # [B, N, C]
        
        # Step 3: Aggregate using mean pooling (simplified - no grouping)
        # Original would group by ix, but we use global mean to avoid scatter
        y_mean = weighted.mean(dim=1, keepdim=True)  # [B, 1, C]
        y = y_mean.expand(B, N, C)  # [B, N, C]
        
        # Step 4: Apply final transformation
        y = self.h(y)  # [B, N, C]
        
        return y



#Successful 2026-01-08 Alister
class SoftAggCV28Exact(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x, ix):
        """
        x  : [B, N, C]
        ix : [N] or [B, N]
        """

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if ix.dim() == 1:
            ix = ix.unsqueeze(0)

        B, N, C = x.shape
        device, dtype = x.device, x.dtype

        # ---- flatten batch (DPVO uses B=1) ----
        ix0 = ix[0]                       # [N]
        fx = self.f(x)[0]
        gx = self.g(x)[0]
        gx = torch.clamp(gx, -50, 50)
        exp_gx = torch.exp(gx)

        # ---- sort by group id ----
        sorted_ix, perm = torch.sort(ix0)
        fx = fx[perm]
        exp_gx = exp_gx[perm]

        # ---- detect group boundaries ----
        is_new = torch.ones_like(sorted_ix)
        is_new[1:] = (sorted_ix[1:] != sorted_ix[:-1]).long()

        group_id = torch.cumsum(is_new, dim=0) - 1
        G = int(group_id[-1].item() + 1)

        # ---- aggregate ----
        denom = torch.zeros(G, C, device=device, dtype=dtype)
        y = torch.zeros(G, C, device=device, dtype=dtype)

        denom.index_add_(0, group_id, exp_gx)
        w = exp_gx / denom[group_id]
        y.index_add_(0, group_id, fx * w)

        y = self.h(y)

        # ---- expand back ----
        out = torch.zeros(N, C, device=device, dtype=y.dtype)

        out[perm] = y[group_id]

        return out.unsqueeze(0)


class SoftAggONNX(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x, ix):
        """
        x  : [1, H, C]
        ix : [1, H] or [H]
        """

        if ix.dim() == 1:
            ix = ix.unsqueeze(0)

        B, H, C = x.shape
        assert B == 1, "DPVO assumes batch size = 1"

        fx = self.f(x)                    # [1, H, C]
        gx = self.g(x)
        gx = torch.clamp(gx, -50, 50)
        w = torch.exp(gx)                 # [1, H, C]

        ix0 = ix[0]                       # [H]

        # ---- equality mask ----
        # mask[h, k] = 1 if ix[h] == ix[k]
        ix_i = ix0.unsqueeze(1)           # [H,1]
        ix_j = ix0.unsqueeze(0)           # [1,H]
        mask = (ix_i == ix_j).to(x.dtype) # [H,H]
        mask = mask.unsqueeze(0).unsqueeze(-1)  # [1,H,H,1]

        # ---- denominator ----
        wj = w.unsqueeze(2)               # [1,H,1,C]
        denom = torch.sum(mask * wj, dim=1)  # [1,H,C]
        denom = torch.clamp(denom, min=1e-9)

        # ---- weighted sum ----
        wi = w / denom                    # [1,H,C]
        fxj = fx.unsqueeze(2)             # [1,H,1,C]
        y = torch.sum(mask * fxj * wi.unsqueeze(2), dim=1)

        return self.h(y)


class SoftAggONNX_ori(nn.Module):
    def __init__(self, dim=512, expand=True):
        super().__init__()
        self.expand = expand
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x, ix):
        """
        x  : [N, C] or [B, N, C]
        ix : [N]  (sparse group indices, e.g., kk/ii/jj)
        Returns:
            - if expand=True: [N, C] or [B, N, C]
            - else: [G, C] or [B, G, C], G = number of unique groups
        """

        is_batched = x.dim() == 3
        device = x.device
        dtype = x.dtype

        # 1️⃣ compress group indices to 0..G-1
        unique_ix, idx_map = torch.unique(ix, return_inverse=True)
        G = unique_ix.numel()

        # 2️⃣ compute projections
        fx = self.f(x)
        gx = self.g(x)

        # 3️⃣ FP16-safe exponential
        if dtype == torch.float16:
            gx = torch.clamp(gx, -50.0, 50.0)
        exp_gx = torch.exp(gx)

        # 4️⃣ prepare group sum and output
        if is_batched:
            B, N, C = x.shape
            denom = torch.zeros(B, G, C, device=device, dtype=dtype)
            y = torch.zeros(B, G, C, device=device, dtype=dtype)

            # iterate over batch dimension (small in DPVO)
            for b in range(B):
                # compute denominator per group
                denom[b].index_add_(0, idx_map, exp_gx[b])
                # compute weighted sum per group
                y[b].index_add_(0, idx_map, fx[b] * (exp_gx[b] / denom[b, idx_map, :]))

        else:  # [N, C]
            N, C = x.shape
            denom = torch.zeros(G, C, device=device, dtype=dtype)
            y = torch.zeros(G, C, device=device, dtype=dtype)
            denom.index_add_(0, idx_map, exp_gx)
            y.index_add_(0, idx_map, fx * (exp_gx / denom[idx_map, :]))

        # 5️⃣ final projection
        y = self.h(y)

        # 6️⃣ expand back if needed
        if self.expand:
            return y[:, idx_map, :] if is_batched else y[idx_map]

        return y


class UpdateONNX(nn.Module):
    def __init__(self, p, export_onnx=False):
        super(UpdateONNX, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM)
        )
        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM)
        )

        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        # 🔥 改成 CV28 friendly aggregation
        self.agg_kk = SoftAggCV28Friendly(DIM)
        self.agg_ij = SoftAggCV28Friendly(DIM)

        self.export_onnx = export_onnx

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2 * 49 * p * p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip()
        )
        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid()
        )

    def forward(
        self,
        net, inp, corr,
        ii, jj, kk,
        kk_idx_map, G_kk,        # 🔥 新增
        ij_idx_map, G_ij         # 🔥 新增
    ):
        # ---------- reshape (CV28 style) ----------
        net  = net.squeeze(-1).permute(0, 2, 1)    # [B, N, C]
        inp  = inp.squeeze(-1).permute(0, 2, 1)
        corr = corr.squeeze(-1).permute(0, 2, 1)

        ii = ii.squeeze(0).squeeze(-1)
        jj = jj.squeeze(0).squeeze(-1)
        kk = kk.squeeze(0).squeeze(-1)

        # ---------- base update ----------
        net = net + inp + self.corr(corr)
        net = self.norm(net)

        # ---------- neighbor indexing ----------
        ix, jx = neighbors_tensor(kk, jj)

        ix = torch.clamp(ix, 0, net.shape[1] - 1)
        jx = torch.clamp(jx, 0, net.shape[1] - 1)

        net = net + self.c1(net[:, ix])
        net = net + self.c2(net[:, jx])

        # ---------- 🔥 soft aggregation (NO Unique) ----------
        net = net + self.agg_kk(net, kk_idx_map, G_kk)
        net = net + self.agg_ij(net, ij_idx_map, G_ij)

        # ---------- GRU ----------
        net = self.gru(net)

        # ---------- outputs ----------
        d_out = self.d(net)
        w_out = self.w(net)

        if self.export_onnx:
            net_out = net.permute(0, 2, 1).unsqueeze(-1)
            d_out = d_out.permute(0, 2, 1).unsqueeze(-1)
            w_out = w_out.permute(0, 2, 1).unsqueeze(-1)
            return net_out, d_out, w_out

        return net, (d_out, w_out, None)



class UpdateONNX_ori(nn.Module):
    def __init__(self, p, export_onnx=False):
        super(UpdateONNX_ori, self).__init__()
        self.c1 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.c2 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.norm = nn.LayerNorm(DIM, eps=1e-3)
        self.agg_kk = SoftAggONNX_ori(DIM)
        self.agg_ij = SoftAggONNX_ori(DIM)
        self.export_onnx = export_onnx
        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )
        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )
        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2), GradientClip())
        self.w = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2), GradientClip(), nn.Sigmoid())

    def forward(self, net, inp, corr, ii, jj, kk):
        # CV28 input reshape (4D -> 3D)
        net  = net.squeeze(-1).permute(0, 2, 1)
        inp  = inp.squeeze(-1).permute(0, 2, 1)
        corr = corr.squeeze(-1).permute(0, 2, 1)

        ii = ii.squeeze(0).squeeze(-1)
        jj = jj.squeeze(0).squeeze(-1)
        kk = kk.squeeze(0).squeeze(-1)

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        # compute neighbors
        ix, jx = neighbors_tensor(kk, jj)

        mask_ix = torch.ones_like(ix, dtype=net.dtype).view(1, -1, 1)  # use full mask
        mask_jx = torch.ones_like(jx, dtype=net.dtype).view(1, -1, 1)

        # clamp indices to [0, H-1] for CV28
        ix = torch.clamp(ix, 0, net.shape[1]-1)
        jx = torch.clamp(jx, 0, net.shape[1]-1)

        # neighbor updates
        net = net + self.c1(net[:, ix])
        net = net + self.c2(net[:, jx])

        # soft aggregation
        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        # outputs
        d_out = self.d(net)
        w_out = self.w(net)

        if self.export_onnx:
            net_out = net.permute(0, 2, 1).unsqueeze(-1)
            d_out = d_out.permute(0, 2, 1).unsqueeze(-1)
            w_out = w_out.permute(0, 2, 1).unsqueeze(-1)
            return net_out, d_out, w_out

        return net, (d_out, w_out, None)


class Update_CV28(nn.Module):
    def __init__(self, p=3, dim=384):
        super().__init__()
        self.dim = dim
        self.p = p

        corr_channels = 2 * 49 * p * p  # 882 for p=3

        # Correlation head
        self.corr = nn.Sequential(
            nn.Linear(corr_channels, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        # Neighbor transforms
        self.c1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.c2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        # Output heads
        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
        )
        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            nn.Sigmoid(),
        )

        # Dummy conv to keep ii/jj/kk as ONNX inputs
        self.keep_conv = nn.Linear(1, 1, bias=False)
        self.keep_conv.weight.data.zero_()
        self.keep_conv.requires_grad_(False)

    def neighbor_slice(self, net):
        """
        Shift neighbor slices for ±1 neighbors along the N dimension.
        net: [B, N, DIM]
        Returns: prev_net, next_net: [B, N, DIM]
        """
        # Pad left for previous neighbor
        prev_net = F.pad(net[:, :-1, :], (0, 0, 1, 0), mode='replicate')
        # Pad right for next neighbor
        next_net = F.pad(net[:, 1:, :], (0, 0, 0, 1), mode='replicate')
        return prev_net, next_net

    def forward(self, net, inp, corr, ii, jj, kk, flow=None):
        """
        Args:
            net: [B, N, DIM]
            inp: [B, N, DIM]
            corr: [B, N, corr_dim]
            ii,jj,kk: [B, N] or [N]
            flow: unused
        Returns:
            net: [B, N, DIM]
            (d_out, w_out, None): d_out/w_out [B, N, 2]
        """
        B, N, DIM = net.shape

        # Dummy op to register ii/jj/kk as inputs in ONNX
        dummy_ii = self.keep_conv(ii.unsqueeze(-1).float())
        dummy_jj = self.keep_conv(jj.unsqueeze(-1).float())
        dummy_kk = self.keep_conv(kk.unsqueeze(-1).float())
        net = net + (dummy_ii + dummy_jj + dummy_kk) * 0.0

        # Correlation addition
        net = net + inp + self.corr(corr)

        # Neighbor slices
        prev_net, next_net = self.neighbor_slice(net)
        net = net + self.c1(prev_net) + self.c2(next_net)

        # Outputs
        d_out = self.d(net)
        w_out = self.w(net)

        return net, (d_out, w_out, None)
