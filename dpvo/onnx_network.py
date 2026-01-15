"""
ONNX-compatible network wrapper for DPVO

This module provides ONNX Runtime-based implementations of Patchifier and Update
that can replace the PyTorch models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import altcorr
from .onnx_inference import ONNXInference

DIM = 384


class ONNXPatchifier(nn.Module):
    """ONNX-based patchifier that uses ONNX models for fnet and inet"""
    
    def __init__(self, patch_size=3, onnx_inference: ONNXInference = None):
        super(ONNXPatchifier, self).__init__()
        self.patch_size = patch_size
        self.onnx_inference = onnx_inference
    
    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
    
    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """
        Extract patches from input images using ONNX models.
        
        This method is compatible with the original Patchifier interface and can be
        used as a drop-in replacement when using ONNX models.
        
        Args:
            images: Input images, shape [B, N, 3, H, W] (typically [1, 1, 3, H, W])
            patches_per_image: Number of patches to extract per image
            disps: Disparity maps (optional)
            centroid_sel_strat: Strategy for patch centroid selection ('RANDOM' or 'GRADIENT_BIAS')
            return_color: Whether to return color information
        
        Returns:
            If return_color=True: (fmap, gmap, imap, patches, index, clr)
            If return_color=False: (fmap, gmap, imap, patches, index)
        """
        # Get device from input images
        device = images.device
        
        # Run fnet and inet through ONNX
        fmap = self.onnx_inference.fnet_forward(images)
        imap = self.onnx_inference.inet_forward(images)
        
        # Ensure correct shape: [B, N, C, H, W]
        if len(fmap.shape) == 4:
            fmap = fmap.unsqueeze(0)  # [N, C, H, W] -> [1, N, C, H, W]
        if len(imap.shape) == 4:
            imap = imap.unsqueeze(0)  # [N, C, H, W] -> [1, N, C, H, W]
        
        b, n, c, h, w = fmap.shape
        P = self.patch_size
        
        # Bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device=device)
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device=device)
            
            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])
        
        elif centroid_sel_strat == 'RANDOM':
            x = torch.randint(1, w-1, size=[n, patches_per_image], device=device)
            y = torch.randint(1, h-1, size=[n, patches_per_image], device=device)
        
        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")
        
        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)
        
        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)
        
        if disps is None:
            disps = torch.ones(b, n, h, w, device=device)
        
        from .utils import coords_grid_with_index
        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)
        
        index = torch.arange(n, device=device).view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)
        
        if return_color:
            return fmap, gmap, imap, patches, index, clr
        
        return fmap, gmap, imap, patches, index


class ONNXUpdate(nn.Module):
    """ONNX-based update module"""
    
    def __init__(self, p, onnx_inference: ONNXInference = None):
        super(ONNXUpdate, self).__init__()
        self.p = p
        self.onnx_inference = onnx_inference
    
    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """
        Forward pass using ONNX update model.
        
        Args:
            net: Network hidden state, shape [1, H, DIM]
            inp: Input features (imap), shape [1, H, DIM]
            corr: Correlation features, shape [1, H, corr_dim]
            flow: Optical flow (unused, kept for compatibility)
            ii: Source frame indices, shape [H]
            jj: Target frame indices, shape [H]
            kk: Patch indices, shape [H]
        
        Returns:
            net: Updated network state, shape [1, H, DIM]
            (delta, weight, None): Tuple containing delta and weight
        """
        return self.onnx_inference.update_forward(net, inp, corr, ii, jj, kk)


class ONNXVONet(nn.Module):
    """ONNX-compatible VONet wrapper"""
    
    def __init__(self, onnx_inference: ONNXInference = None):
        super(ONNXVONet, self).__init__()
        self.P = 3
        self.patchify = ONNXPatchifier(self.P, onnx_inference)
        self.update = ONNXUpdate(self.P, onnx_inference)
        self.DIM = DIM
        self.RES = 4

