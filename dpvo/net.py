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
        if not ONNX_COMPATIBLE:
            net = net + inp + self.corr(corr)
            net = self.norm(net)

            # ix, jx = fastba.neighbors(kk, jj)
            ix, jx = neighbors_tensor(kk, jj)
        
            mask_ix = (ix >= 0).float().reshape(1, -1, 1)
            mask_jx = (jx >= 0).float().reshape(1, -1, 1)

            net = net + self.c1(mask_ix * net[:,ix])
            net = net + self.c2(mask_jx * net[:,jx])

            net = net + self.agg_kk(net, kk)
            net = net + self.agg_ij(net, ii*12345 + jj)

            net = self.gru(net)
            
            return net, (self.d(net), self.w(net), None)
        else:
            # CRITICAL: Ensure ii is used early to prevent ONNX from optimizing it away
            # Convert to float and create a dependency that affects the computation
            ii_float = ii.float()  # Convert to float for operations
            # Create a minimal bias tensor from ii that gets added to net
            # Use a very small but non-zero value to create dependency chain
            ii_bias = (ii_float.sum() * 1e-10).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            ii_bias = ii_bias.expand_as(net)  # [1, H, DIM]
            # Add to net - this creates a dependency that ONNX cannot optimize away
            net = net + inp + self.corr(corr) + ii_bias
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
            ii_used = ii.to(torch.int64)  # Ensure ii is processed
            ii_used = torch.clamp(ii_used, min=0, max=net.shape[1]-1)  # Clamp to valid range
            
            net = net + self.agg_kk(net, kk)
            # Use ii_used instead of ii directly to create a dependency chain
            net = net + self.agg_ij(net, ii_used*12345 + jj)

            net = self.gru(net)

            # outputs
            d_out = self.d(net)
            w_out = self.w(net)
            
            # CRITICAL: Ensure ii affects outputs to prevent ONNX from optimizing it away
            # Add minimal contribution based on ii to create dependency chain
            # This ensures ONNX sees ii as required for the outputs
            ii_final = ii.float().unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
            ii_scale = 1e-10  # Very small but non-zero to create dependency
            
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

    return ix.to(torch.int64), jx.to(torch.int64)



class SoftAggONNX(nn.Module):
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
        self.c1 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.c2 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.norm = nn.LayerNorm(DIM, eps=1e-3)
        self.agg_kk = SoftAggONNX(DIM)
        self.agg_ij = SoftAggONNX(DIM)
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
