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

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

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


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """
        # Residual Fusion 
        net = net + inp + self.corr(corr)
        # Layer Norm
        net = self.norm(net)
        # Gives the indices of neighboring nodes/patches for each element in kk and jj. 
        # These are then used to perform neighbor-based updates in the network.
        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)
        
        # Neighbor Update:  
        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        # Soft Aggregation:   
        net = net + self.agg_kk(net, kk)
        # net = net + self.agg_ij(net, ii*12345 + jj)
        agg_ix = torch.clamp(ii*12345 + jj, 0, net.shape[1]-1)
        net = net + self.agg_ij(net, agg_ix)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)


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


#=====================================================================================================================
# Alister add 2025-12-15 for convert update model into onnx

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



def neighbors_tensor_v2(ii: torch.Tensor, jj: torch.Tensor):
    """
    Fully tensorized neighbors computation WITHOUT torch.unique (ONNX-compatible)
    ii, jj: any shape -> flattened to [N]
    Returns:
        ix, jx: previous and next neighbor indices, shape [N]
    """
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    N = ii.shape[0]

    LARGE = 1_000_000
    SMALL = -1_000_000

    # Broadcast ii and jj to compare all pairs
    ii_row = ii.unsqueeze(0).expand(N, N)
    ii_col = ii.unsqueeze(1).expand(N, N)
    jj_row = jj.unsqueeze(0).expand(N, N)
    jj_col = jj.unsqueeze(1).expand(N, N)

    # Mask to only allow neighbors in the same group
    mask = ii_row == ii_col  # [N, N]

    # Previous neighbor: jj_row < jj_col
    prev_mask = mask & (jj_row < jj_col)
    prev_jj = torch.where(prev_mask, jj_row, torch.full_like(jj_row, SMALL))
    ix = torch.argmax(prev_jj, dim=1)
    ix = torch.where(prev_mask.any(dim=1), ix, torch.full_like(ix, -1))

    # Next neighbor: jj_row > jj_col
    next_mask = mask & (jj_row > jj_col)
    next_jj = torch.where(next_mask, jj_row, torch.full_like(jj_row, LARGE))
    jx = torch.argmin(next_jj, dim=1)
    jx = torch.where(next_mask.any(dim=1), jx, torch.full_like(jx, -1))

    return ix.to(torch.int64), jx.to(torch.int64)


def neighbors_tensor_v1(ii: torch.Tensor, jj: torch.Tensor):
    """
    Fully tensorized neighbors computation (ONNX-compatible)
    ii, jj: any shape -> flattened to [N]
    """

    # ✅ CRITICAL FIX: flatten indices
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    N = ii.shape[0]

    uniq, inv = torch.unique(ii, sorted=True, return_inverse=True)
    segment_mask = inv.unsqueeze(0) == inv.unsqueeze(1)  # [N, N]

    jj_row = jj.unsqueeze(0).expand(N, N)
    jj_col = jj.unsqueeze(1).expand(N, N)

    LARGE = 1_000_000
    SMALL = -1_000_000

    jj_row = torch.where(segment_mask, jj_row, torch.full_like(jj_row, LARGE))

    prev_mask = jj_row < jj.unsqueeze(1)
    prev_jj = torch.where(prev_mask, jj_row, torch.full_like(jj_row, SMALL))
    ix = torch.argmax(prev_jj, dim=1)
    ix = torch.where(prev_mask.any(dim=1), ix, torch.full_like(ix, -1))

    next_mask = jj_row > jj.unsqueeze(1)
    next_jj = torch.where(next_mask, jj_row, torch.full_like(jj_row, LARGE))
    jx = torch.argmin(next_jj, dim=1)
    jx = torch.where(next_mask.any(dim=1), jx, torch.full_like(jx, -1))

    return ix.to(torch.int64), jx.to(torch.int64)


class UpdateONNX(nn.Module):
    def __init__(self, p,export_onnx=False):
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

    
    def forward_v1(self, net, inp, corr, ii, jj, kk):
        """
        net  : [B, C, H, 1]   -> CV28 input
        inp  : [B, C, H, 1]
        corr : [B, Cc, H, 1]
        ii,jj,kk: [1, H]
        """
        """
        Current:
        ii,jj,kk: [1, H]
        New:
        ii,jj,kk: [1, H, 1]
        """
        # ---------------------------
        # CV28 input reshape (4D -> 3D)
        # ---------------------------
        # net:  [B, C, H, 1] -> [B, H, C]
        net  = net.squeeze(-1).permute(0, 2, 1)
        inp  = inp.squeeze(-1).permute(0, 2, 1)
        corr = corr.squeeze(-1).permute(0, 2, 1)

        # indices: [1, H] -> [H]
        ii = ii.squeeze(0).squeeze(-1)  # [H]
        jj = jj.squeeze(0).squeeze(-1)  # [H]
        kk = kk.squeeze(0).squeeze(-1)  # [H]

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        # compute neighbors fully tensorized
        ix, jx = neighbors_tensor(kk, jj)

        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        # neighbor updates
        net = net + self.c1(mask_ix * net[:, ix])
        net = net + self.c2(mask_jx * net[:, jx])

        # soft aggregation
        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        # outputs
        d_out = self.d(net)
        w_out = self.w(net)

        if self.export_onnx:
            # net: [B, H, DIM]
            net_out = net.permute(0, 2, 1).unsqueeze(-1)   # [B, DIM, H, 1]
            # d_out: [B, H, 2] → [B, 2, H, 1]
            d_out = d_out.permute(0, 2, 1).unsqueeze(-1)
            # w_out: [B, H, 2] → [B, 2, H, 1]
            w_out = w_out.permute(0, 2, 1).unsqueeze(-1)

            return net_out, d_out, w_out


        return net, (d_out, w_out, None)

#-------------------------------------------------------------------------------------------------------------------
def neighbors(ii: torch.Tensor, jj: torch.Tensor):
    """
    Python version of the original C++ fastba.neighbors()

    Args:
        ii: LongTensor [N], group/landmark indices
        jj: LongTensor [N], sort key (e.g., time/frame index)

    Returns:
        ix, jx: LongTensor [N]
            ix[n] = index of previous neighbor in same group, -1 if none
            jx[n] = index of next neighbor in same group, -1 if none
    """
    """
    Example Usage:
    ii = torch.tensor([0,0,1,1,0,2,2,2,1,0])
    jj = torch.tensor([5,1,2,3,4,1,2,0,5,0])

    ix, jx = neighbors(ii, jj)
    print("ix:", ix)
    print("jx:", jx)
    """
    device = ii.device
    N = ii.shape[0]

    # 1️⃣ Compute unique groups and permutation to map back
    uniq, perm = torch.unique(ii, return_inverse=True)
    # uniq: [num_groups], perm[n] = group index
    num_groups = uniq.shape[0]

    # 2️⃣ Build list of indices per group
    index = [[] for _ in range(num_groups)]
    perm_cpu = perm.cpu()
    for n in range(N):
        index[perm_cpu[n].item()].append(n)

    # 3️⃣ Allocate ix/jx
    ix = torch.empty(N, dtype=torch.long, device=device)
    jx = torch.empty(N, dtype=torch.long, device=device)

    # 4️⃣ Process each group
    jj_cpu = jj.cpu()
    for group_idx in range(num_groups):
        idx_list = index[group_idx]
        # Sort by jj values
        sorted_idx = sorted(idx_list, key=lambda i: jj_cpu[i].item())
        # Fill ix/jx
        for k, i in enumerate(sorted_idx):
            ix[i] = sorted_idx[k-1] if k > 0 else -1
            jx[i] = sorted_idx[k+1] if k < len(sorted_idx)-1 else -1

    # 5️⃣ Move back to original device (GPU/CPU)
    ix = ix.to(device)
    jx = jx.to(device)

    return ix, jx


class UpdateONNX_v1(nn.Module):
    export_onnx = True
    def __init__(self, p):
        super(UpdateONNX_v1, self).__init__()

        # Linear blocks
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

        # Use your ONNX-compatible SoftAgg
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
            GradientClip()
        )

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid()
        )

    def forward(self, net, inp, corr, flow, ii, jj, ix, jx, export_onnx=False):
        """
        net   : [B, N, D]
        inp   : [B, N, D]
        corr  : [B, N, 2*49*p*p]
        ii, jj: [N]  (used for SoftAggONNX)
        ix, jx: [N]  precomputed neighbor indices, -1 means no neighbor
        """

        # --- Residual Fusion ---
        net = net + inp + self.corr(corr)

        # --- Layer Norm ---
        net = self.norm(net)

        # --- Gather neighbors safely for ONNX ---
        B, N, D = net.shape

        # For previous neighbors
        ix_safe = ix.clamp(min=0)  # replace -1 with 0
        net_ix = net[:, ix_safe, :]  # gather
        mask_ix = (ix >= 0).float().view(1, -1, 1)  # [1, N, 1]
        net = net + self.c1(mask_ix * net_ix)

        # For next neighbors
        jx_safe = jx.clamp(min=0)
        net_jx = net[:, jx_safe, :]
        mask_jx = (jx >= 0).float().view(1, -1, 1)
        net = net + self.c2(mask_jx * net_jx)

        # --- Soft Aggregation ---
        net = net + self.agg_kk(net, ii)
        net = net + self.agg_ij(net, ii*12345 + jj)

        # --- GRU blocks ---
        net = self.gru(net)

        # --- Outputs ---
        if export_onnx:
            return net, self.d(net), self.w(net)
        else:
            return net, (self.d(net), self.w(net), None)
       

# ----------------------------------------
# 2️⃣ ONNX-compatible SoftAgg 2025-12-16
# ---------------------------------------

class SoftAggONNX(nn.Module):
    def __init__(self, dim=512, expand=False):
        super().__init__()
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)
        self.expand = expand

    def forward(self, x, ix):
        """
        x  : [B, N, C]
        ix : [N]   int32 (group index)
        """
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        # --------------------------
        # 1️⃣ Ensure int64 indices and clamp
        # --------------------------
        jx = ix.to(torch.int64)
        jx = torch.clamp(jx, min=0, max=N-1)

        # --------------------------
        # 2️⃣ Compute logits
        # --------------------------
        logits = self.g(x)
        exp_logits = torch.exp(logits)

        # --------------------------
        # 3️⃣ Preallocate denominator with static shape [B, N, C]
        # --------------------------
        denom = torch.zeros((B, N, C), device=device, dtype=dtype)
        jx_expand = jx.view(1, N, 1).expand(B, N, C)
        jx_expand = torch.clamp(jx_expand, 0, N-1)
        denom.scatter_add_(1, jx_expand, exp_logits)

        # safe division
        w = exp_logits / torch.clamp(denom, min=1e-6)

        # --------------------------
        # 4️⃣ Weighted scatter sum
        # --------------------------
        fx = self.f(x)
        weighted = fx * w
        y = torch.zeros((B, N, C), device=device, dtype=dtype)
        y.scatter_add_(1, jx_expand, weighted)

        # --------------------------
        # 5️⃣ Output
        # --------------------------
        y = self.h(y)

        if self.expand:
            # gather back to original positions
            y = torch.gather(y, 1, jx_expand)

        return y



class SoftAggONNX_v3(nn.Module):
    def __init__(self, dim=512, expand=True):
        super().__init__()
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)
        self.expand = expand

    def forward(self, x, ix):
        """
        x  : [B, N, C]
        ix : [N]   int32 (group index)
        """
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        # --------------------------
        # 1️⃣ Ensure int64 indices
        # --------------------------
        jx = ix.to(torch.int64)
        jx = torch.clamp(jx, min=0, max=N-1)  # ✅ clamp for CV28

        # --------------------------
        # 2️⃣ Compute logits
        # --------------------------
        logits = self.g(x)
        exp_logits = torch.exp(logits)

        # --------------------------
        # 3️⃣ Preallocate denominator with static shape [B, N, C]
        # --------------------------
        denom = torch.zeros((B, N, C), device=device, dtype=dtype)

        # scatter_add to denominator
        jx_expand = jx.view(1, N, 1).expand(B, N, C)
        denom.scatter_add_(1, jx_expand, exp_logits)

        # safe division
        w = exp_logits / torch.clamp(denom, min=1e-6)

        # --------------------------
        # 4️⃣ Weighted scatter sum
        # --------------------------
        fx = self.f(x)
        weighted = fx * w

        y = torch.zeros((B, N, C), device=device, dtype=dtype)
        y.scatter_add_(1, jx_expand, weighted)

        # --------------------------
        # 5️⃣ Output
        # --------------------------
        y = self.h(y)

        if self.expand:
            # gather back to original positions
            y = torch.gather(y, 1, jx_expand)

        return y



class SoftAggONNX_v2(nn.Module):
    def __init__(self, dim=512, expand=True):
        super().__init__()
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)
        self.expand = expand

    def forward(self, x, ix):
        """
        x : [B, N, C]
        ix: [N]   (group index, int32 from CV28-friendly input)
        """
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        # --------------------------------------------------
        # 1️⃣ FORCE int64 index (PyTorch / ONNX requirement)
        # --------------------------------------------------
        jx = ix.to(torch.int64)   # ✅ REQUIRED

        M = jx.max() + 1          # tensor scalar (int64)

        # --------------------------------------------------
        # 2️⃣ Compute logits
        # --------------------------------------------------
        logits = self.g(x)
        exp_logits = torch.exp(logits)

        # --------------------------------------------------
        # 3️⃣ Group-wise softmax
        # --------------------------------------------------
        denom = torch.zeros((B, M, C), device=device, dtype=dtype)

        jx_expand = jx.view(1, N, 1).expand(B, N, C)

        denom.scatter_add_(1, jx_expand, exp_logits)

        w = exp_logits / torch.gather(denom, 1, jx_expand)

        # --------------------------------------------------
        # 4️⃣ Weighted scatter sum
        # --------------------------------------------------
        fx = self.f(x)
        weighted = fx * w

        y = torch.zeros((B, M, C), device=device, dtype=dtype)
        y.scatter_add_(1, jx_expand, weighted)

        # --------------------------------------------------
        # 5️⃣ Output
        # --------------------------------------------------
        y = self.h(y)

        if self.expand:
            return torch.gather(y, 1, jx_expand)

        return y




class SoftAggONNX_v1(nn.Module):
    def __init__(self, dim=384, expand=True):
        super().__init__()
        self.expand = expand
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x, ii):
        """
        x : [B, N, D]
        ii: [N] (group index, same for all batches)
        Label meaning : 
            Letter	Meaning
            b	    batch
            n	    point index
            m	    group (segment)
            d	    feature dim
        """

        B, N, D = x.shape
        # Remove int(..), This way, M is a tensor and ONNX will handle it dynamically.
        M = ii.max() + 1

        # one-hot group mask
        mask = F.one_hot(ii, M).float()          # [N, M]
        mask = mask.unsqueeze(0)                 # [1, N, M]

        # attention logits
        logits = self.g(x)                       # [B, N, D]
        exp_logits = torch.exp(logits)

        # sum exp per group (batch-wise)
        # Multiply mask[b, n, m] * exp_logits[b, n, d]
        # Sum over n (because n is missing in output)
        '''
        multiply → sum over named dimensions
        sum_exp[b, m, d] = sum over n in group m of exp_logits[b, n, d]
        denom[b, n, d] = sum_exp[b, ii[n], d]
        '''
        sum_exp = torch.einsum("bnm,bnd->bmd", mask, exp_logits)

        # normalize
        # Multiply mask[b,n,m] * sum_exp[b,m,d]
        # Sum over m
        # Result shape [B, N, D]
        w = exp_logits / (torch.einsum("bnm,bmd->bnd", mask, sum_exp) + 1e-6)

        # weighted sum
        y = torch.einsum("bnm,bnd->bmd", mask, self.f(x) * w)

        y = self.h(y)

        if self.expand:
            return y[:, ii]

        return y
