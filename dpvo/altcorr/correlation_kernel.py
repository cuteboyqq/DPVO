import torch
import math
import torch.nn.functional as F
import numpy as np
'''
---------------------------
Patchify forward / backward
---------------------------
'''

'''
--------------------------------------
Forward kernel and wrapper function
-------------------------------------
'''
def patchify_forward_kernel_CPU(R, net, coords):
    """
    Python equivalent of patchify_forward_kernel.

    Args:
        R      : patch radius
        net    : [B, C, H, W] feature map
        coords : [B, M, 2] float coordinates (x, y)

    Returns:
        patches: [B, M, C, D, D]
    """

    B, C, H, W = net.shape
    _, M, _ = coords.shape

    D = 2 * R + 2  # diameter
    patches = torch.zeros((B, M, C, D, D), dtype=net.dtype, device=net.device)

    for n in range(B):               # batch
        for m in range(M):           # coordinate index
            x = coords[n, m, 0].item()
            y = coords[n, m, 1].item()

            base_i = int(math.floor(y))
            base_j = int(math.floor(x))

            for ii in range(D):      # patch row
                for jj in range(D):  # patch col

                    i = base_i + (ii - R)
                    j = base_j + (jj - R)

                    # bounds check
                    if 0 <= i < H and 0 <= j < W:
                        patches[n, m, :, ii, jj] = net[n, :, i, j]

    return patches


def patchify_forward_kernel_python(R, net, coords):
    """
    Python equivalent of patchify_forward_kernel.

    Args:
        R      : patch radius
        net    : [B, C, H, W] feature map
        coords : [B, M, 2] float coordinates (x, y)

    Returns:
        patches: [B, M, C, D, D]
    """
    B, C, H, W = net.shape
    _, M, _ = coords.shape
    D = 2*R + 2
    device = net.device

    flat = net.reshape(-1)

    ii = torch.arange(D, device=device).view(1, 1, D, 1)
    jj = torch.arange(D, device=device).view(1, 1, 1, D)

    y0 = torch.floor(coords[..., 1]).long().view(B, M, 1, 1)
    x0 = torch.floor(coords[..., 0]).long().view(B, M, 1, 1)

    iy = (y0 + (ii - R)).clamp(0, H-1)
    ix = (x0 + (jj - R)).clamp(0, W-1)

    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
    c_idx = torch.arange(C, device=device).view(1, C, 1, 1)

    idx = (
        b_idx * (C*H*W) +
        c_idx * (H*W) +
        iy.unsqueeze(2) * W +
        ix.unsqueeze(2)
    )

    idx_flat = idx.reshape(-1)

    patches_flat = flat.index_select(0, idx_flat)
    patches = patches_flat.reshape(B, M, C, D, D)

    return patches



def patchify_python_forward(net, coords, radius):
    B, C, H, W = net.shape
    M = coords.shape[1]
    D = 2 * radius + 2

    patches = patchify_forward_kernel_python(radius, net, coords)

    return [patches]

'''
-------------------------------------
Backward kernel and wrapper function
-------------------------------------
'''
def patchify_backward_kernel(R, patch_gradient, coords, H, W):
    """
    Vectorized patchify backward (gradient w.r.t input feature map).

    Args:
        R               : int, radius
        patch_gradient  : [B, M, C, D, D]   (gradient w.r.t patches)
        coords          : [B, M, 2]         (x,y)
        H, W            : int, input feature map size

    Returns:
        gradient : [B, C, H, W]  (gradient w.r.t input feature map)
    """
    B, M, C, D, _ = patch_gradient.shape
    device = patch_gradient.device
    gradient = torch.zeros(B, C, H, W, dtype=patch_gradient.dtype, device=device)

    # 1) Base coordinates
    x0 = torch.floor(coords[..., 0]).long()  # [B, M]
    y0 = torch.floor(coords[..., 1]).long()  # [B, M]

    # 2) Offsets
    ii = torch.arange(D, device=device).view(1, 1, D, 1)
    jj = torch.arange(D, device=device).view(1, 1, 1, D)

    # 3) Compute sample positions in the input gradient
    iy = (y0.view(B, M, 1, 1) + (ii - R)).clamp(0, H-1)  # [B, M, D, D]
    ix = (x0.view(B, M, 1, 1) + (jj - R)).clamp(0, W-1)  # [B, M, D, D]

    # 4) Expand for channels
    c_idx = torch.arange(C, device=device).view(1, C, 1, 1, 1, 1)  # [1,C,1,1,1,1]
    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1, 1)  # [B,1,1,1,1,1]

    # iy, ix expand for channel dimension
    iy = iy.unsqueeze(1).expand(B, C, M, D, D)  # [B,C,M,D,D]
    ix = ix.unsqueeze(1).expand(B, C, M, D, D)
    patch_grad = patch_gradient.permute(0, 2, 1, 3, 4)  # [B,C,M,D,D]

    # 5) Flatten all dimensions for scatter_add
    gradient = gradient.view(B, C, H*W)
    index = iy * W + ix  # [B,C,M,D,D]
    gradient.scatter_add_(2, index.view(B, C, -1), patch_grad.view(B, C, -1))

    # 6) reshape back
    gradient = gradient.view(B, C, H, W)
    return gradient


def patchify_python_backward(net, coords, patch_grad, radius):
    """
    Python equivalent of patchify_cuda_backward wrapper.
    Calls patchify_backward (Python version of patchify_backward_kernel).
    
    Args:
        net        : [B, C, H, W] input feature map
        coords     : [B, M, 2] coordinates of patches
        patch_grad : [B, M, C, D, D] gradient w.r.t patches
        radius     : int, patch radius
    
    Returns:
        list containing one tensor: net_gradient [B, C, H, W]
    """
    B, C, H, W = net.shape
    D = 2 * radius + 2

    # Call the Python backward kernel
    net_gradient = patchify_backward_kernel(radius, patch_grad, coords, H, W)

    return [net_gradient]

'''
-------------------------------
Correlation forward & backward
--------------------------------
'''


'''
---------
Forward
---------
'''

def corr_forward_center_only(fmap1, fmap2, coords, ii, jj, batch_size=512):
    """
    Compute correlation ONLY at the center point (no search window).
    Memory-efficient batched version.
    
    Args:
        fmap1: [B, N, C, H, W] first feature maps
        fmap2: [B, N, C, H2, W2] second feature maps
        coords: [B, M, 2, H, W] coordinates (x, y)
        ii: [M] indices for fmap1
        jj: [M] indices for fmap2
        batch_size: process M in batches
    
    Returns:
        corr: [B, M, H, W] correlation at center point only
    """
    B, N, C, H, W = fmap1.shape
    _, M, _, _, _ = coords.shape
    _, _, _, H2, W2 = fmap2.shape
    
    device = fmap1.device
    dtype = fmap1.dtype
    
    # Extract coordinates
    x = coords[:, :, 0]  # [B, M, H, W]
    y = coords[:, :, 1]  # [B, M, H, W]
    
    # Pre-select all f1 features (this is manageable)
    ii_exp = ii.view(1, M, 1, 1, 1).expand(B, M, C, H, W)
    f1_all = torch.gather(fmap1, 1, ii_exp)  # [B, M, C, H, W]
    
    # Initialize output
    corr = torch.zeros(B, M, H, W, dtype=dtype, device=device)
    
    # Process in batches
    for m_start in range(0, M, batch_size):
        m_end = min(m_start + batch_size, M)
        m_batch = m_end - m_start
        
        # Get batch of jj indices and find unique ones
        jj_batch = jj[m_start:m_end]
        unique_jj, inverse_idx = torch.unique(jj_batch, return_inverse=True)
        
        # Load only unique f2 features needed
        f2_unique = fmap2[:, unique_jj]  # [B, n_unique, C, H2, W2]
        
        # Get batch data
        f1_batch = f1_all[:, m_start:m_end]  # [B, m_batch, C, H, W]
        x_batch = x[:, m_start:m_end]  # [B, m_batch, H, W]
        y_batch = y[:, m_start:m_end]  # [B, m_batch, H, W]
        
        # Map to unique f2 indices
        # inverse_idx tells us which unique_jj each m in the batch corresponds to
        f2_batch_list = [f2_unique[:, inverse_idx[i]] for i in range(m_batch)]
        f2_batch = torch.stack(f2_batch_list, dim=1)  # [B, m_batch, C, H2, W2]
        
        # Reshape for grid_sample
        f1_flat = f1_batch.reshape(B * m_batch, C, H, W)
        f2_flat = f2_batch.reshape(B * m_batch, C, H2, W2)
        x_flat = x_batch.reshape(B * m_batch, H, W)
        y_flat = y_batch.reshape(B * m_batch, H, W)
        
        # Normalize coordinates to [-1, 1] for grid_sample
        x_norm = 2.0 * x_flat / (W2 - 1) - 1.0
        y_norm = 2.0 * y_flat / (H2 - 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1)  # [B*m_batch, H, W, 2]
        
        # Sample f2 at exact coordinates
        f2_sampled = F.grid_sample(f2_flat, grid, mode='bilinear', 
                                   padding_mode='zeros', align_corners=True)
        
        # Compute correlation
        corr_batch = (f1_flat * f2_sampled).sum(dim=1)  # [B*m_batch, H, W]
        corr_batch = corr_batch.reshape(B, m_batch, H, W)
        
        # Store
        corr[:, m_start:m_end] = corr_batch
    
    return corr


def corr_forward_center_only_padded(fmap1, fmap2, coords, ii, jj, radius, batch_size=512):
    """
    Compute correlation at center and pad to match full window output shape.
    This allows using the fast center-only computation with existing networks
    that expect full window shape.
    
    Args:
        fmap1: [B, N, C, H, W] first feature maps
        fmap2: [B, N, C, H2, W2] second feature maps
        coords: [B, M, 2, H, W] coordinates (x, y)
        ii: [M] indices for fmap1
        jj: [M] indices for fmap2
        radius: search radius (used only for output shape)
        batch_size: process M in batches
    
    Returns:
        corr: [B, M, H, D, W, D] with only center filled, rest zeros
              where D = 2*radius+1
    """
    D = 2 * radius + 1
    center_idx = radius  # Middle of the window
    
    # Compute center correlation
    corr_center = corr_forward_center_only(fmap1, fmap2, coords, ii, jj, batch_size)
    # corr_center shape: [B, M, H, W]
    
    B, M, H, W = corr_center.shape
    device = corr_center.device
    dtype = corr_center.dtype
    
    # Create output with full window shape
    corr_full = torch.zeros(B, M, D, D, H, W, dtype=dtype, device=device)
    
    # Place center correlation in the middle of the window
    corr_full[:, :, center_idx, center_idx, :, :] = corr_center
    
    # Permute to match expected output format [B, M, H, D, W, D]
    corr_full = corr_full.permute(0, 1, 4, 2, 5, 3)
    
    return corr_full




def corr_forward_fast(fmap1, fmap2, coords, ii, jj, radius, patch_batch=128):
    """
    Memory-safe DPVO-style patch correlation.

    Inputs:
      fmap1: (B, N, C, ph, pw)   # stacked patches, e.g., 3456 = 96*36
      fmap2: (B, S, C, H2, W2)   # feature maps to correlate against
      coords: (B, M, 2, ph, pw)  # float coordinates (x,y) per patch
      ii: (M,)                   # indices into fmap1
      jj: (M,)                   # indices into fmap2
      radius: int
      patch_batch: number of patches per sub-batch

    Returns:
      corr: (B, M, ph, D, pw, D)
    """
    B, N, C, ph, pw = fmap1.shape
    _, S, C2, H2, W2 = fmap2.shape
    _, M, two, ph2, pw2 = coords.shape
    D = 2 * radius + 1
    device = fmap1.device
    dtype = fmap1.dtype

    assert two == 2 and ph2 == ph and pw2 == pw, "coords shape mismatch"

    corr_all = torch.empty(B, M, ph, D, pw, D, device=device, dtype=dtype)

    # Build offset grid in DPVO order: y outer, x inner
    dy = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    dx = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(dy, dx, indexing='ij')  # (D, D)
    ox = ox.contiguous().view(1, 1, 1, -1)  # (1,1,1,K)
    oy = oy.contiguous().view(1, 1, 1, -1)
    K = D*D
    batch_idx = torch.arange(B, device=device)[:, None]

    for m_start in range(0, M, patch_batch):
        m_end = min(m_start + patch_batch, M)
        m_size = m_end - m_start

        # Gather indices
        ii_batch = ii[m_start:m_end].to(device)
        jj_batch = jj[m_start:m_end].to(device)
        coords_batch = coords[:, m_start:m_end].to(device)  # (B, m_size, 2, ph, pw)

        # Gather fmap1/fmap2 patches (memory-safe)
        fmap1_sel = fmap1[batch_idx, ii_batch[None,:].expand(B, m_size)]  # (B, m_size, C, ph, pw)
        fmap2_sel = fmap2[batch_idx, jj_batch[None,:].expand(B, m_size)]  # (B, m_size, C, H2, W2)

        # Flatten patch positions
        Kp = ph*pw
        x_base = coords_batch[:, :, 0].reshape(B, m_size, Kp, 1)
        y_base = coords_batch[:, :, 1].reshape(B, m_size, Kp, 1)

        # Add offset grid
        x_shifts = x_base + ox  # (B, m_size, Kp, K)
        y_shifts = y_base + oy

        # Bilinear sampling
        x0 = torch.floor(x_shifts).long().clamp(0, W2-1)
        x1 = (x0 + 1).clamp(max=W2-1)
        y0 = torch.floor(y_shifts).long().clamp(0, H2-1)
        y1 = (y0 + 1).clamp(max=H2-1)

        wx = (x_shifts - x0.to(dtype))
        wy = (y_shifts - y0.to(dtype))
        wa = (1-wx)*(1-wy)
        wb = wx*(1-wy)
        wc = (1-wx)*wy
        wd = wx*wy

        fmap2_flat = fmap2_sel.reshape(B, m_size, C, H2*W2)
        def gather(idx):
            idx_e = idx.reshape(B, m_size, 1, Kp*K).expand(B, m_size, C, Kp*K)
            return torch.gather(fmap2_flat, dim=3, index=idx_e)

        idx00 = (y0*W2 + x0)
        idx10 = (y0*W2 + x1)
        idx01 = (y1*W2 + x0)
        idx11 = (y1*W2 + x1)

        v00 = gather(idx00)
        v10 = gather(idx10)
        v01 = gather(idx01)
        v11 = gather(idx11)

        wa_r = wa.reshape(B, m_size, 1, Kp*K)
        wb_r = wb.reshape(B, m_size, 1, Kp*K)
        wc_r = wc.reshape(B, m_size, 1, Kp*K)
        wd_r = wd.reshape(B, m_size, 1, Kp*K)

        sampled = wa_r*v00 + wb_r*v10 + wc_r*v01 + wd_r*v11
        sampled = sampled.reshape(B, m_size, C, Kp, K)  # (B, m_size, C, Kp, K)

        # Compute correlation
        f1_flat = fmap1_sel.reshape(B, m_size, C, Kp)
        corr_kp_k = (f1_flat.unsqueeze(-1) * sampled).sum(dim=2)  # (B, m_size, Kp, K)
        corr_patch = corr_kp_k.reshape(B, m_size, ph, pw, D, D)
        # Permute to DPVO layout: (B, M, ph, D, pw, D)
        corr_all[:, m_start:m_end] = corr_patch.permute(0,1,2,4,3,5).contiguous()

    return corr_all




def corr_forward_with_window_safe(fmap1, fmap2, coords, ii, jj, radius, batch_size=4):
    """
    Memory-safe version of correlation with D×D search window around each coordinate.
    It processes M in small batches and loops over each (dy,dx) offset — only one sampled
    feature map is held in memory at a time.

    Args:
        fmap1: [B, N, C, H, W] first feature maps
        fmap2: [B, N, C, H2, W2] second feature maps
        coords: [B, M, 2, H, W] exact coordinates (x, y)
        ii: [M] indices for fmap1
        jj: [M] indices for fmap2
        radius: search radius
        batch_size: number of M entries processed in each loop (keep small to save memory)

    Returns:
        corr: [B, M, H, D, W, D] correlation volume with bilinear sampling
              (same output shape/permute order as your original function)
    """
    D = 2 * radius + 1
    B, N, C, H, W = fmap1.shape
    _, M, _, _, _ = coords.shape
    _, _, _, H2, W2 = fmap2.shape

    device = fmap1.device
    dtype = fmap1.dtype

    # basic sanity
    assert coords.shape == (B, M, 2, H, W)
    assert ii.numel() == M and jj.numel() == M

    # Pre-select all f1 features: gather fmap1 by ii -> [B, M, C, H, W]
    # ii may be CPU tensor; ensure correct device for gather view/expand
    ii_exp = ii.view(1, M, 1, 1, 1).expand(B, M, C, H, W).to(device)
    f1_all = torch.gather(fmap1, 1, ii_exp)  # [B, M, C, H, W]

    # Extract coordinate channels
    x = coords[:, :, 0]  # [B, M, H, W]
    y = coords[:, :, 1]  # [B, M, H, W]

    # Prepare output: [B, M, D, D, H, W]
    corr = torch.zeros(B, M, D, D, H, W, dtype=dtype, device=device)

    # Precompute offset list (integer offsets)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

    # Process in m-batches
    for m_start in range(0, M, batch_size):
        m_end = min(m_start + batch_size, M)
        m_batch = m_end - m_start

        # slice
        jj_batch = jj[m_start:m_end].to(device)
        unique_jj, inverse_idx = torch.unique(jj_batch, return_inverse=True)
        # f2_unique: [B, U, C, H2, W2]
        f2_unique = fmap2[:, unique_jj]  # gather along N

        # Build f2_batch list by mapping inverse_idx -> [B, m_batch, C, H2, W2]
        # This avoids expanding to D*D simultaneously.
        # We will reshape when needed to [B*m_batch, C, H2, W2], which is OK for small m_batch.
        f2_batch_list = [f2_unique[:, inverse_idx[i]] for i in range(m_batch)]
        f2_batch = torch.stack(f2_batch_list, dim=1)  # [B, m_batch, C, H2, W2]

        # f1 batch: [B, m_batch, C, H, W]
        f1_batch = f1_all[:, m_start:m_end]

        # coords batch: [B, m_batch, H, W]
        x_batch = x[:, m_start:m_end]
        y_batch = y[:, m_start:m_end]

        # flatten channel dims for grid_sample usage
        # We'll create f1_flat once (it is small: H*W)
        f1_flat = f1_batch.reshape(B * m_batch, C, H, W)  # [B*m_batch, C, H, W]
        f2_flat = f2_batch.reshape(B * m_batch, C, H2, W2)  # [B*m_batch, C, H2, W2]
        # Note: these two reshapes are OK as long as batch_size (=> m_batch) is small.

        # To speed up repeated grid_sample calls, we will create f2_flat once and reuse it.
        # For each offset, build the grid, sample f2_flat, then compute dot with f1_flat and accumulate.

        # Precompute base normalized multipliers
        # normalization: x_norm = 2.0 * x / (W2 - 1) - 1.0
        norm_x_scale = 2.0 / (W2 - 1)
        norm_y_scale = 2.0 / (H2 - 1)

        # Reshape coordinate batches to [B*m_batch, H, W]
        x_base = x_batch.reshape(B * m_batch, H, W)
        y_base = y_batch.reshape(B * m_batch, H, W)

        # For each offset, sample and accumulate
        offset_idx = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # shifted coordinates (float) in f2 pixel space
                x_shift = x_base + float(dx)     # [B*m_batch, H, W]
                y_shift = y_base + float(dy)     # [B*m_batch, H, W]

                # normalize to [-1, 1]
                x_norm = x_shift * norm_x_scale - 1.0
                y_norm = y_shift * norm_y_scale - 1.0

                # build grid for grid_sample: [B*m_batch, H, W, 2] with (x, y) ordering
                # grid_sample expects last dim (x, y)
                grid = torch.stack([x_norm, y_norm], dim=-1)  # [B*m_batch, H, W, 2]

                # sample f2: result [B*m_batch, C, H, W]
                # Use align_corners=True to match original code (was using align_corners=True earlier)
                f2_sampled = F.grid_sample(f2_flat, grid, mode='bilinear',
                                           padding_mode='zeros', align_corners=True)

                # elementwise multiply with f1_flat and sum over channel -> [B*m_batch, H, W]
                corr_offset = (f1_flat * f2_sampled).sum(dim=1)

                # reshape back to [B, m_batch, H, W] and place into corr
                corr_batch_offset = corr_offset.reshape(B, m_batch, H, W)

                # store at corr[:, m_start:m_end, dy_idx, dx_idx, :, :]
                dy_idx = dy + radius
                dx_idx = dx + radius
                corr[:, m_start:m_end, dy_idx, dx_idx, :, :] = corr_batch_offset

                offset_idx += 1

        # end offsets loop

    # end m-batches loop

    # Permute to [B, M, H, D, W, D] to match original function's final layout
    corr = corr.permute(0, 1, 4, 2, 5, 3).contiguous()
    return corr


def corr_forward_with_window(fmap1, fmap2, coords, ii, jj, radius, batch_size=256):
    """
    Compute correlation with D×D search window around each coordinate.
    Memory-efficient batched version.
    
    Args:
        fmap1: [B, N, C, H, W] first feature maps
        fmap2: [B, N, C, H2, W2] second feature maps
        coords: [B, M, 2, H, W] exact coordinates (x, y) 
        ii: [M] indices for fmap1
        jj: [M] indices for fmap2
        radius: search radius
        batch_size: process M in batches
    
    Returns:
        corr: [B, M, H, D, W, D] correlation volume with bilinear sampling
    """
    D = 2 * radius + 1
    B, N, C, H, W = fmap1.shape
    _, M, _, _, _ = coords.shape
    _, _, _, H2, W2 = fmap2.shape
    
    device = fmap1.device
    dtype = fmap1.dtype
    
    # Extract coordinates
    x = coords[:, :, 0]  # [B, M, H, W]
    y = coords[:, :, 1]  # [B, M, H, W]
    
    # Pre-select all f1 features
    ii_exp = ii.view(1, M, 1, 1, 1).expand(B, M, C, H, W)
    f1_all = torch.gather(fmap1, 1, ii_exp)  # [B, M, C, H, W]
    
    # Create offset grid
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    
    # Initialize output
    corr = torch.zeros(B, M, D, D, H, W, dtype=dtype, device=device)
    
    # Process in batches
    for m_start in range(0, M, batch_size):
        m_end = min(m_start + batch_size, M)
        m_batch = m_end - m_start
        
        # Get batch
        jj_batch = jj[m_start:m_end]
        unique_jj, inverse_idx = torch.unique(jj_batch, return_inverse=True)
        f2_unique = fmap2[:, unique_jj]
        
        f1_batch = f1_all[:, m_start:m_end]
        x_batch = x[:, m_start:m_end]
        y_batch = y[:, m_start:m_end]
        
        # Map to unique f2
        f2_batch_list = [f2_unique[:, inverse_idx[i]] for i in range(m_batch)]
        f2_batch = torch.stack(f2_batch_list, dim=1)  # [B, m_batch, C, H2, W2]
        
        # Add offsets: [B, m_batch, H, W] -> [B, m_batch, D, D, H, W]
        # x_offset = x_batch.unsqueeze(2).unsqueeze(3) + offsets.view(1, 1, 1, D, 1, 1)
        # y_offset = y_batch.unsqueeze(2).unsqueeze(3) + offsets.view(1, 1, D, 1, 1, 1)
        
        oy, ox = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=device),
            torch.arange(-radius, radius + 1, device=device),
            indexing="ij"
        )

        x_offset = x_batch[:, :, None, None] + ox[None, None, :, :, None, None]
        y_offset = y_batch[:, :, None, None] + oy[None, None, :, :, None, None]

        
        # Reshape for grid_sample
        # x_offset, y_offset are [B, m_batch, D, D, H, W]
        BmD2 = B * m_batch * D * D
        
        x_grid = x_offset.reshape(BmD2, H, W)
        y_grid = y_offset.reshape(BmD2, H, W)
        
        # Normalize
        x_norm = 2.0 * x_grid / (W2 - 1) - 1.0
        y_norm = 2.0 * y_grid / (H2 - 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1)
        
        # Expand f1 and f2
        f1_exp = f1_batch.unsqueeze(2).unsqueeze(3).expand(B, m_batch, D, D, C, H, W)
        f1_flat = f1_exp.reshape(BmD2, C, H, W)
        
        f2_exp = f2_batch.unsqueeze(2).unsqueeze(3).expand(B, m_batch, D, D, C, H2, W2)
        f2_flat = f2_exp.reshape(BmD2, C, H2, W2)
        
        # Sample and correlate
        f2_sampled = F.grid_sample(f2_flat, grid, mode='bilinear', 
                                   padding_mode='zeros', align_corners=True)
        
        corr_batch = (f1_flat * f2_sampled).sum(dim=1)  # [BmD2, H, W]
        corr_batch = corr_batch.reshape(B, m_batch, D, D, H, W)
        
        # Store
        corr[:, m_start:m_end] = corr_batch
    
    # Permute to [B, M, H, D, W, D]
    corr = corr.permute(0, 1, 4, 2, 5, 3)
    
    return corr


def corr_cuda_forward(fmap1, fmap2, coords, ii, jj, radius, batch_size=1, use_center_only=False):
    """
    Compute correlation with search window and bilinear interpolation.
    Matches the original CUDA implementation output format.
    
    Args:
        fmap1: [B, N, C, H, W] first feature maps
        fmap2: [B, N, C, H2, W2] second feature maps  
        coords: [B, M, 2, H, W] coordinates (x, y)
        ii: [M] indices for fmap1
        jj: [M] indices for fmap2
        radius: search radius
        batch_size: number of M to process at once (adjust based on GPU memory)
        use_center_only: if True, only compute center (fast) and pad to full shape.
                        Network sees same shape but only center has real values.
    
    Returns:
        [B, M, H, D, W, D] where D = 2*radius+1
    """
    if use_center_only:
        return corr_forward_center_only_padded(fmap1, fmap2, coords, ii, jj, radius, batch_size)
    else:
        return corr_forward_fast(fmap1, fmap2, coords, ii, jj, radius)



def corr_forward_kernel_ver0(R, fmap1, fmap2, coords, us, vs, use_half=True):
    """
    Memory-safe correlation forward kernel for DPVO.

    Args:
        R: correlation radius
        fmap1: [B, N1, C, H1, W1]
        fmap2: [B, N2, C, H2, W2]
        coords: [B, M, 2, H, W]
        us: patch indices for fmap1 [M]
        vs: patch indices for fmap2 [M]
        use_half: if True, use float16 internally for memory savings

    Returns:
        corr: [B, M, D, D, H, W]
    """
    D = 2 * R + 2
    B, M, _, H, W = coords.shape
    C = fmap1.size(2)
    H2, W2 = fmap2.size(3), fmap2.size(4)
    device = fmap1.device
    dtype = torch.float16 if use_half else fmap1.dtype

    # Floor coordinates and convert to long
    x = coords[:, :, 0, :, :].floor().long()
    y = coords[:, :, 1, :, :].floor().long()

    # Initialize output tensor
    corr = torch.zeros(B, M, D, D, H, W, device=device, dtype=dtype)

    ii_offsets = torch.arange(D, device=device) - R
    jj_offsets = torch.arange(D, device=device) - R

    for b in range(B):
        x_b, y_b = x[b], y[b]

        for m in range(M):
            f1_m = fmap1[b, us[m]].to(dtype)  # [C,H,W]
            f2_m = fmap2[b, vs[m]].to(dtype)  # [C,H2,W2]
            x_m, y_m = x_b[m], y_b[m]

            for i in range(D):
                i_idx = (y_m + ii_offsets[i]).clamp(0, H2-1)
                for j in range(D):
                    j_idx = (x_m + jj_offsets[j]).clamp(0, W2-1)
                    mask = ((y_m + ii_offsets[i] >= 0) & (y_m + ii_offsets[i] < H2) &
                            (x_m + jj_offsets[j] >= 0) & (x_m + jj_offsets[j] < W2)).float()

                    f2_vals = f2_m[:, i_idx, j_idx]  # [C,H,W]
                    corr[b, m, i, j] = (f1_m * f2_vals).sum(0) * mask

    return corr


# def corr_cuda_forward(fmap1, fmap2, coords, ii, jj, radius, use_half=True):
#     """
#     Memory-efficient correlation forward wrapper.
#     Supports optional mixed precision.
#     """
#     D = 2 * radius + 2

#     # Compute correlation
#     corr = corr_forward_kernel_chunked(radius, fmap1, fmap2, coords, ii, jj, use_half=use_half)

#     # Bilinear interpolation
#     x = coords[:, :, 0, :, :]
#     y = coords[:, :, 1, :, :]
#     dx = (x - x.floor()).to(corr.dtype).unsqueeze(2).unsqueeze(3)  # [B,M,1,1,H,W]
#     dy = (y - y.floor()).to(corr.dtype).unsqueeze(2).unsqueeze(3)

#     out = (1 - dx) * (1 - dy) * corr[:, :, 0:D-1, 0:D-1, :, :]
#     out += dx * (1 - dy) * corr[:, :, 0:D-1, 1:D, :, :]
#     out += (1 - dx) * dy * corr[:, :, 1:D, 0:D-1, :, :]
#     out += dx * dy * corr[:, :, 1:D, 1:D, :, :]

#     # Permute to match DPVO expected shape
#     out = out.permute(0, 1, 3, 2, 5, 4)

#     return [out]



def corr_forward_kernel_ver1(R, fmap1, fmap2, coords, us, vs):
    """
    ONNX-exportable vectorized version of corr_forward_kernel.

    Args:
        R      : int, search radius
        fmap1  : [B, U, C, H, W]  (torch.Tensor)
        fmap2  : [B, V, C, H2, W2] (torch.Tensor)
        coords : [B, M, 2, H, W]   (float coordinates; coords[b,m,0,i0,j0]=x, coords[b,m,1,i0,j0]=y)
        us     : [M] (long) -> indices into dim=1 of fmap1 (U)
        vs     : [M] (long) -> indices into dim=1 of fmap2 (V)

    Returns:
        corr   : [B, M, D, D, H, W]  (same ordering as kernel: corr[n][m][ii][jj][i0][j0])
    """
    device = fmap1.device
    dtype = fmap1.dtype

    B, U, C, H, W = fmap1.shape
    _, V, C2, H2, W2 = fmap2.shape
    assert C2 == C, "channel mismatch"
    _, M, two, Hc, Wc = coords.shape
    assert two == 2 and Hc == H and Wc == W, "coords shape must be [B, M, 2, H, W]"
    assert H == Hc and W == Wc

    D = 2 * R + 2

    # 1) flatten input feature volumes for ONNX-safe gather (index_select)
    flat1 = fmap1.reshape(-1)  # length = B*U*C*H*W
    flat2 = fmap2.reshape(-1)  # length = B*V*C*H2*W2

    # 2) compute base (floor) coordinates per [B, M, H, W]
    # coords: [B, M, 2, H, W]
    x = coords[:, :, 0, :, :]  # [B, M, H, W]
    y = coords[:, :, 1, :, :]  # [B, M, H, W]
    y0 = torch.floor(y).long()  # [B, M, H, W]
    x0 = torch.floor(x).long()  # [B, M, H, W]

    # 3) build search-grid offsets
    ii = torch.arange(D, device=device, dtype=torch.long).view(1, 1, D, 1, 1)  # [1,1,D,1,1]
    jj = torch.arange(D, device=device, dtype=torch.long).view(1, 1, 1, D, 1)  # [1,1,1,D,1]

    # i1, j1 positions in fmap2 for each offset: [B, M, D, D, H, W]
    iy = (y0.view(B, M, 1, 1, H, W) + (ii - R)).clamp(0, H2 - 1)  # [B,M,D,D,H,W]
    ix = (x0.view(B, M, 1, 1, H, W) + (jj - R)).clamp(0, W2 - 1)  # [B,M,D,D,H,W]

    # 4) create indices for gathering from flattened fmap1 and fmap2
    # Index formula (linear index into flattened arrays):
    # for fmap1: idx1 = b*(U*C*H*W) + u*(C*H*W) + c*(H*W) + i0*W + j0
    # for fmap2: idx2 = b*(V*C*H2*W2) + v*(C*H2*W2) + c*(H2*W2) + i1*W2 + j1

    # Prepare base multipliers (long)
    stride_b_1 = (U * C * H * W)
    stride_u = (C * H * W)
    stride_c_1 = (H * W)

    stride_b_2 = (V * C * H2 * W2)
    stride_v = (C * H2 * W2)
    stride_c_2 = (H2 * W2)

    # Expand indices shapes:
    # b_idx: [B, 1, 1, 1, 1, 1]
    b_idx = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1, 1, 1, 1)

    # us and vs are [M], make [1, M, 1, 1, 1, 1]
    us = us.to(device=device).long().view(1, M, 1, 1, 1, 1)
    vs = vs.to(device=device).long().view(1, M, 1, 1, 1, 1)

    # c_idx: [1, 1, C, 1, 1, 1]
    c_idx = torch.arange(C, device=device, dtype=torch.long).view(1, 1, C, 1, 1, 1)

    # i0,j0 coordinates (original grid) -> shape [B, M, 1, 1, H, W]
    i0 = torch.arange(H, device=device, dtype=torch.long).view(1, 1, 1, 1, H, 1)
    j0 = torch.arange(W, device=device, dtype=torch.long).view(1, 1, 1, 1, 1, W)

    # Build idx1: [B, M, C, D?,D?, H, W] - but fmap1 doesn't have D dims, so we will expand later
    # idx1_base shape [B, M, C, H, W]
    idx1_base = (
        b_idx * stride_b_1 +               # [B,1,1,1,1,1] broadcast
        us * stride_u +                    # [1,M,1,1,1,1]
        c_idx * stride_c_1 +               # [1,1,C,1,1,1]
        i0 * W +                           # [1,1,1,1,H,1]
        j0                                 # [1,1,1,1,1,W]
    )  # result broadcasts to [B, M, C, H, W]

    # Now idx2: needs D,D dims and uses iy,ix: shape [B, M, C, D, D, H, W]
    # Compose idx2 similarly:
    # Note: ensure iy and ix are long
    iy_long = iy.long()  # [B,M,D,D,H,W]
    ix_long = ix.long()

    # Expand to include channel dimension at dim=2
    # b_idx2: [B,1,1,1,1,1,1]
    b_idx2 = b_idx  # already [B,1,1,1,1,1]
    # reshape multipliers to broadcast when needed
    # compute idx2:
    # idx2 = b*(V*C*H2*W2) + vs*(C*H2*W2) + c*(H2*W2) + iy*W2 + ix
    # shapes:
    # b_idx2 -> [B,1,1,1,1,1]
    # vs -> [1,M,1,1,1,1]
    # c_idx -> [1,1,C,1,1,1]
    # iy_long -> [B,M,D,D,H,W] -> need to insert dim for C at pos=2
    iy_c = iy_long.unsqueeze(2)   # [B,M,1,D,D,H,W]
    ix_c = ix_long.unsqueeze(2)   # [B,M,1,D,D,H,W]

    # Now compute idx2 with broadcasting into [B,M,C,D,D,H,W]
    idx2 = (
        b_idx * stride_b_2 +
        vs * stride_v +
        c_idx * stride_c_2 +
        iy_c * W2 +
        ix_c
    )  # broadcasts to [B,M,C,D,D,H,W]

    # Expand idx1_base to include D,D dims to match idx2 shape:
    # idx1_base: [B,M,C,H,W] -> want [B,M,C,D,D,H,W] by unsqueeze & expand
    idx1 = idx1_base.unsqueeze(3).unsqueeze(3)  # [B,M,C,1,1,H,W]
    idx1 = idx1.expand(B, M, C, D, D, H, W)

    # At this point:
    # idx1: [B,M,C,D,D,H,W]  (long)
    # idx2: [B,M,C,D,D,H,W]  (long)

    # 5) flatten and gather from flattened feature vectors
    idx1_flat = idx1.reshape(-1)  # 1D index vector
    idx2_flat = idx2.reshape(-1)

    vals1_flat = flat1.index_select(0, idx1_flat)  # 1D tensor of length L = B*M*C*D*D*H*W
    vals2_flat = flat2.index_select(0, idx2_flat)

    # reshape back to [B, M, C, D, D, H, W]
    vals1 = vals1_flat.view(B, M, C, D, D, H, W)
    vals2 = vals2_flat.view(B, M, C, D, D, H, W)

    # 6) multiply and sum across channel dimension (dim=2) -> result [B, M, D, D, H, W]
    prod = vals1 * vals2
    corr = torch.sum(prod, dim=2)  # sum over C

    return corr




def corr_backward_kernel(radius, fmap1, fmap2, coords, ii, jj, corr_grad, fmap1_grad, fmap2_grad):
    """
    Vectorized PyTorch version of corr_backward_kernel.
    Accumulates gradients into fmap1_grad and fmap2_grad without Python loops.
    """
    B, M, _, H, W = coords.shape
    C = fmap1.shape[1]
    D = 2 * radius + 2

    # coords: [B, M, 2, H, W]
    x = coords[:, :, 0]  # [B, M, H, W]
    y = coords[:, :, 1]

    # create grid indices
    ii_idx = torch.arange(D, device=coords.device).view(1,1,D,1,1)
    jj_idx = torch.arange(D, device=coords.device).view(1,1,1,D,1,1)

    i0 = torch.arange(H, device=coords.device).view(1,1,1,1,H,1)
    j0 = torch.arange(W, device=coords.device).view(1,1,1,1,1,W)

    # expand batch and M dims
    b_idx = torch.arange(B, device=coords.device).view(B,1,1,1,1,1)
    m_idx = torch.arange(M, device=coords.device).view(1,M,1,1,1,1)

    # compute source/dest indices
    ix = ii[m_idx]  # [1, M, 1, 1, 1, 1] -> broadcast
    jx = jj[m_idx]

    # compute shifted coordinates
    i1 = (y[:, :, None, None, :, :] + (ii_idx - radius)).long().clamp(0, fmap1.shape[2]-1)
    j1 = (x[:, :, None, None, :, :] + (jj_idx - radius)).long().clamp(0, fmap1.shape[3]-1)

    # flatten for gather/scatter
    B_M_DD_HW = B * M * D * D * H * W
    corr_grad_flat = corr_grad.reshape(B, M, D, D, H, W)

    for c in range(C):
        # fmap1_grad[b, ix, c, i0, j0] += g * fmap2[b, jx, c, i1, j1]
        fmap1_grad.index_put_(
            (b_idx.expand_as(corr_grad), ix.expand_as(corr_grad), c, i0.expand_as(corr_grad), j0.expand_as(corr_grad)),
            corr_grad_flat * fmap2[:, :, c][:, :, None, None, :, :],
            accumulate=True
        )

        # fmap2_grad[b, jx, c, i1, j1] += g * fmap1[b, ix, c, i0, j0]
        fmap2_grad.index_put_(
            (b_idx.expand_as(corr_grad), jx.expand_as(corr_grad), c, i1, j1),
            corr_grad_flat * fmap1[:, :, c][:, :, None, None, :, :],
            accumulate=True
        )


def corr_cuda_backward(fmap1, fmap2, coords, ii, jj, grad, radius):
    """
    Wrapper function: prepares corr_grad and calls vectorized kernel.
    """
    B, M, _, H, W = coords.shape
    D = 2 * radius + 2

    # permute grad to match original CUDA indexing
    grad = grad.permute(0,1,3,2,4,5).contiguous()  # [B, M, D, D, H, W]

    # bilinear weights
    x = coords[:, :, 0:1]
    y = coords[:, :, 1:2]
    dx = x - torch.floor(x)
    dy = y - torch.floor(y)

    g1 = torch.zeros((B, M, D, D, H, W), device=grad.device, dtype=grad.dtype)
    g2 = torch.zeros_like(g1)
    g3 = torch.zeros_like(g1)
    g4 = torch.zeros_like(g1)

    g1[:, :, 0:D-1, 0:D-1] = (1 - dx) * (1 - dy) * grad
    g2[:, :, 0:D-1, 1:D]   = dx * (1 - dy) * grad
    g3[:, :, 1:D,   0:D-1] = (1 - dx) * dy * grad
    g4[:, :, 1:D,   1:D]   = dx * dy * grad

    corr_grad = g1 + g2 + g3 + g4

    fmap1_grad = torch.zeros_like(fmap1)
    fmap2_grad = torch.zeros_like(fmap2)

    corr_backward_kernel(radius, fmap1, fmap2, coords, ii, jj, corr_grad, fmap1_grad, fmap2_grad)

    return fmap1_grad, fmap2_grad
