import torch
import math
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl

'''
-------------------------------
corr using Trition
-------------------------------
'''
import torch
import triton
import triton.language as tl

# --------------------------
# Triton kernel
# --------------------------
import torch
import triton
import triton.language as tl

# --------------------------
# Triton kernel
# --------------------------
@triton.jit
def corr_triton_kernel_safe(
    fmap1_ptr, fmap2_ptr, coords_ptr, ii_ptr, jj_ptr, corr_ptr,
    B: tl.constexpr, M: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    H2: tl.constexpr, W2: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    stride_fmap1_b, stride_fmap1_m, stride_fmap1_c, stride_fmap1_h, stride_fmap1_w,
    stride_fmap2_b, stride_fmap2_m, stride_fmap2_c, stride_fmap2_h, stride_fmap2_w,
    stride_coords_b, stride_coords_m, stride_coords_c, stride_coords_h, stride_coords_w,
    stride_corr_b, stride_corr_m, stride_corr_r, stride_corr_rj, stride_corr_h, stride_corr_w
):
    pid = tl.program_id(0)
    j0 = pid % W; pid //= W
    i0 = pid % H; pid //= H
    m  = pid % M; pid //= M
    b  = pid % B

    # Load memory indices
    ix = tl.load(ii_ptr + m)
    jx = tl.load(jj_ptr + m)

    # Load coordinates
    x = tl.load(coords_ptr + b*stride_coords_b + m*stride_coords_m + 0*stride_coords_c + i0*stride_coords_h + j0*stride_coords_w)
    y = tl.load(coords_ptr + b*stride_coords_b + m*stride_coords_m + 1*stride_coords_c + i0*stride_coords_h + j0*stride_coords_w)

    # Iterate over patch window
    for ii in range(0, 2*R+1):
        for jj in range(0, 2*R+1):
            i1 = tl.cast(tl.floor(y) + (ii - R), tl.int32)
            j1 = tl.cast(tl.floor(x) + (jj - R), tl.int32)

            # Bounds mask
            mask = (i1 >= 0) & (i1 < H2) & (j1 >= 0) & (j1 < W2)

            s = 0.0
            if mask:
                for c_start in range(0, C, BLOCK_C):
                    offs = tl.arange(0, BLOCK_C)
                    valid = (offs + c_start) < C

                    # Load fmap1
                    f1_offset = (c_start + offs)*stride_fmap1_c + i0*stride_fmap1_h + j0*stride_fmap1_w
                    f1 = tl.load(fmap1_ptr + b*stride_fmap1_b + ix*stride_fmap1_m + f1_offset, mask=valid)

                    # Load fmap2
                    f2_offset = (c_start + offs)*stride_fmap2_c + i1*stride_fmap2_h + j1*stride_fmap2_w
                    f2 = tl.load(fmap2_ptr + b*stride_fmap2_b + jx*stride_fmap2_m + f2_offset, mask=valid)

                    s += tl.sum(f1 * f2)

            # Store result safely
            corr_offset = ii*stride_corr_r + jj*stride_corr_rj + i0*stride_corr_h + j0*stride_corr_w
            tl.store(corr_ptr + b*stride_corr_b + m*stride_corr_m + corr_offset, s, mask=mask)


# --------------------------
# Python wrapper
# --------------------------
def corr_triton_forward_safe(fmap1, fmap2, coords, ii, jj, radius, BLOCK_C=32):
    """
    Safe Triton correlation forward pass.
    fmap1: [B, M, C, H, W]
    fmap2: [B, M, C, H2, W2]
    coords: [B, M, 2, H, W]
    ii, jj: [M]
    radius: int
    """
    # Ensure contiguous
    fmap1 = fmap1.contiguous()
    fmap2 = fmap2.contiguous()
    coords = coords.contiguous()
    ii = ii.contiguous()
    jj = jj.contiguous()

    B, M, C, H, W = fmap1.shape
    _, _, _, H2, W2 = fmap2.shape
    D = 2*radius + 1

    corr = torch.zeros((B, M, D, D, H, W), device=fmap1.device, dtype=fmap1.dtype)

    # Strides
    stride_fmap1_b, stride_fmap1_m, stride_fmap1_c, stride_fmap1_h, stride_fmap1_w = fmap1.stride()
    stride_fmap2_b, stride_fmap2_m, stride_fmap2_c, stride_fmap2_h, stride_fmap2_w = fmap2.stride()
    stride_coords_b, stride_coords_m, stride_coords_c, stride_coords_h, stride_coords_w = coords.stride()
    stride_corr_b, stride_corr_m, stride_corr_r, stride_corr_rj, stride_corr_h, stride_corr_w = corr.stride()

    grid = (B*M*H*W, )

    corr_triton_kernel_safe[grid](
        fmap1, fmap2, coords, ii, jj, corr,
        B, M, H, W, H2, W2, C, radius,
        BLOCK_C,
        stride_fmap1_b, stride_fmap1_m, stride_fmap1_c, stride_fmap1_h, stride_fmap1_w,
        stride_fmap2_b, stride_fmap2_m, stride_fmap2_c, stride_fmap2_h, stride_fmap2_w,
        stride_coords_b, stride_coords_m, stride_coords_c, stride_coords_h, stride_coords_w,
        stride_corr_b, stride_corr_m, stride_corr_r, stride_corr_rj, stride_corr_h, stride_corr_w
    )

    # Return corr tensor directly (bilinear interpolation can be done in PyTorch if needed)
    return corr



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
import math
def within_bounds(h, w, H, W):
    return (h >= 0) and (h < H) and (w >= 0) and (w < W)

def corr_forward_torch(R, fmap1, fmap2, coords, us, vs):
    """
    Python reference implementation of corr_forward_kernel

    Args:
        R: int
        fmap1: (B, ?, C, H, W)
        fmap2: (B, ?, C, H2, W2)
        coords: (B, M, 2, H, W)
        us: (M,)
        vs: (M,)

    Returns:
        corr: (B, M, D, D, H, W)
    """

    device = fmap1.device
    dtype = fmap1.dtype

    D = 2 * R + 2

    B, M, _, H, W = coords.shape
    C = fmap1.shape[2]
    H2, W2 = fmap2.shape[3], fmap2.shape[4]

    corr = torch.zeros((B, M, D, D, H, W), device=device, dtype=dtype)

    for b in range(B):
        for m in range(M):
            ix = int(us[m].item())
            jx = int(vs[m].item())

            for i0 in range(H):
                for j0 in range(W):

                    x = coords[b, m, 0, i0, j0].item()
                    y = coords[b, m, 1, i0, j0].item()

                    base_i = math.floor(y)
                    base_j = math.floor(x)

                    for ii in range(D):
                        for jj in range(D):

                            i1 = base_i + (ii - R)
                            j1 = base_j + (jj - R)

                            if within_bounds(i1, j1, H2, W2):
                                # dot product over channel C
                                f1 = fmap1[b, ix, :, i0, j0]
                                f2 = fmap2[b, jx, :, i1, j1]
                                s = torch.dot(f1, f2)
                            else:
                                s = torch.tensor(0.0, device=device, dtype=dtype)

                            corr[b, m, ii, jj, i0, j0] = s

    return corr


def corr_forward_torch_wrapper(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    coords: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    radius: int,
):
    """
    Python translation of corr_cuda_forward

    Args:
        fmap1: (B, ?, C, H, W)
        fmap2: (B, ?, C, H2, W2)
        coords: (B, M, 2, H, W)
        ii: (M,)
        jj: (M,)
        radius: int

    Returns:
        [out]: list with one tensor
        out shape: (B, M, D-1, D-1, H, W) permuted
    """

    B, M, _, H, W = coords.shape
    D = 2 * radius + 2

    # ---------------------------------------------------------
    # 1. Call correlation kernel (Python version)
    # ---------------------------------------------------------
    corr = corr_forward_torch(
        R=radius,
        fmap1=fmap1,
        fmap2=fmap2,
        coords=coords,
        us=ii,
        vs=jj,
    )
    # corr shape: (B, M, D, D, H, W)

    # ---------------------------------------------------------
    # 2. Extract x, y and fractional offsets
    # ---------------------------------------------------------
    x = coords[:, :, 0, None, None]   # (B, M, 1, 1, H, W)
    y = coords[:, :, 1, None, None]

    dx = x - torch.floor(x)
    dy = y - torch.floor(y)

    dx = dx.to(fmap1.dtype)
    dy = dy.to(fmap2.dtype)

    # ---------------------------------------------------------
    # 3. Bilinear interpolation on correlation volume
    # ---------------------------------------------------------
    out = (
        (1 - dx) * (1 - dy) * corr[:, :, 0:D-1, 0:D-1]
        + (dx)     * (1 - dy) * corr[:, :, 0:D-1, 1:D]
        + (1 - dx) * (dy)     * corr[:, :, 1:D,   0:D-1]
        + (dx)     * (dy)     * corr[:, :, 1:D,   1:D]
    )

    # ---------------------------------------------------------
    # 4. Permute to match DPVO layout
    # ---------------------------------------------------------
    out = out.permute(0, 1, 3, 2, 4, 5)

    # ---------------------------------------------------------
    # 5. Match C++ return type
    # ---------------------------------------------------------
    return [out]
'''==============================================================================================================================='''

def corr_torch_forward(
    fmap1,
    fmap2,
    coords,
    ii,
    jj,
    radius,
    chunk_size=64,   # << key parameter
):  
    # fmap1 = fmap1.half()
    # fmap2 = fmap2.half()
    # coords = coords.half()

    device = fmap1.device
    dtype = fmap1.dtype

    B, M, _, H, W = coords.shape
    C = fmap1.size(2)
    H2, W2 = fmap2.size(3), fmap2.size(4)
    D = 2 * radius + 2

    # output
    corr = torch.empty((B, M, D, D, H, W), device=device, dtype=dtype)

    # offsets
    offs = torch.arange(-radius, radius + 2, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offs, offs, indexing='ij')  # [D,D]

    ox = ox.view(1, 1, D, D, 1, 1)
    oy = oy.view(1, 1, D, D, 1, 1)

    
    for m0 in range(0, M, chunk_size):
        m1 = min(m0 + chunk_size, M)
        mc = m1 - m0

        ii_c = ii[m0:m1]
        jj_c = jj[m0:m1]

        f1 = fmap1[:, ii_c]        # [B, mc, C, H, W]
        f2 = fmap2[:, jj_c]        # [B, mc, C, H2, W2]

        x = coords[:, m0:m1, 0]
        y = coords[:, m0:m1, 1]

        x0 = torch.floor(x)
        y0 = torch.floor(y)

        # grid
        gx = x0.unsqueeze(2).unsqueeze(2) + ox
        gy = y0.unsqueeze(2).unsqueeze(2) + oy

        gx = 2 * gx / (W2 - 1) - 1
        gy = 2 * gy / (H2 - 1) - 1

        grid = torch.stack([gx, gy], dim=-1)
        grid = grid.view(B * mc, D * D * H * W, 1, 2)

        f2 = f2.view(B * mc, C, H2, W2)

        sampled = F.grid_sample(
            f2, grid,
            mode='bilinear',
            align_corners=True
        )

        sampled = sampled.view(B, mc, C, D, D, H, W)

        # dot
        corr[:, m0:m1] = (
            f1.unsqueeze(3).unsqueeze(3) * sampled
        ).sum(dim=2)

    # ---- wrapper bilinear interpolation ----
    dx = coords[:, :, 0] - torch.floor(coords[:, :, 0])
    dy = coords[:, :, 1] - torch.floor(coords[:, :, 1])

    dx = dx.unsqueeze(2).unsqueeze(2)
    dy = dy.unsqueeze(2).unsqueeze(2)

    out = (
        (1 - dx) * (1 - dy) * corr[:, :, 0:D-1, 0:D-1]
        + dx * (1 - dy)     * corr[:, :, 0:D-1, 1:D]
        + (1 - dx) * dy     * corr[:, :, 1:D, 0:D-1]
        + dx * dy           * corr[:, :, 1:D, 1:D]
    )

    return out.permute(0, 1, 3, 2, 4, 5)



def corr_torch_forward_fp16(
    fmap1, fmap2, coords, ii, jj, radius, chunk_size=512
):
    """
    FP16, chunked, GPU-friendly correlation for DPVO.
    Returns: [B, M, D-1, D-1, H, W]
    """
    device = fmap1.device
    dtype = torch.half  # Force FP16

    B, M, _, H, W = coords.shape
    C = fmap1.size(2)
    H2, W2 = fmap2.size(3), fmap2.size(4)
    D = 2 * radius + 2

    # Ensure tensors are FP16
    fmap1 = fmap1.half()
    fmap2 = fmap2.half()
    coords = coords.half()
    # output
    corr = torch.empty((B, M, D, D, H, W), device=device, dtype=dtype)

    # offsets
    offs = torch.arange(-radius, radius + 2, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offs, offs, indexing='ij')  # [D,D]
    ox = ox.view(1, 1, D, D, 1, 1)
    oy = oy.view(1, 1, D, D, 1, 1)

    # process in chunks
    # for m0 in range(0, M, chunk_size):
    
    MAX_PATCHES = 10000
    m_start = max(0, M - MAX_PATCHES)

    for m0 in range(m_start, M, chunk_size):
        m1 = min(m0 + chunk_size, M)
        
        mc = m1 - m0

        ii_c = ii[m0:m1]
        jj_c = jj[m0:m1]

        f1 = fmap1[:, ii_c]        # [B, mc, C, H, W]
        f2 = fmap2[:, jj_c]        # [B, mc, C, H2, W2]

        x = coords[:, m0:m1, 0]
        y = coords[:, m0:m1, 1]

        x0 = torch.floor(x)
        y0 = torch.floor(y)

        # grid for sampling
        gx = x0.unsqueeze(2).unsqueeze(2) + ox  # [B, mc, D, D, H, W]
        gy = y0.unsqueeze(2).unsqueeze(2) + oy

        gx = 2 * gx / (W2 - 1) - 1
        gy = 2 * gy / (H2 - 1) - 1

        grid = torch.stack([gx, gy], dim=-1).view(B * mc, D * D * H * W, 1, 2)
        f2_view = f2.view(B * mc, C, H2, W2)

        sampled = F.grid_sample(
            f2_view, grid, mode='bilinear', align_corners=True
        )  # [B*mc, C, D*D*H*W, 1]

        sampled = sampled.view(B, mc, C, D, D, H, W)

        # dot product over channels
        f1e = f1.unsqueeze(3).unsqueeze(3)  # [B, mc, C, 1, 1, H, W]
        corr[:, m0:m1] = (f1e * sampled).sum(dim=2)

    # ---- wrapper bilinear interpolation ----
    dx = (coords[:, :, 0] - torch.floor(coords[:, :, 0])).unsqueeze(2).unsqueeze(2)
    dy = (coords[:, :, 1] - torch.floor(coords[:, :, 1])).unsqueeze(2).unsqueeze(2)

    dx = dx.half()
    dy = dy.half()

    # out = (
    #     (1 - dx) * (1 - dy) * corr[:, :, 0:D-1, 0:D-1]
    #     + dx * (1 - dy)     * corr[:, :, 0:D-1, 1:D]
    #     + (1 - dx) * dy     * corr[:, :, 1:D, 0:D-1]
    #     + dx * dy           * corr[:, :, 1:D, 1:D]
    # )
    out = torch.zeros(
        (B, M, D-1, D-1, H, W),
        device=corr.device,
        dtype=corr.dtype
    )
    # only update recent patches
    out[:, m_start:M] = (
        (1 - dx[:, m_start:M]) * (1 - dy[:, m_start:M]) * corr[:, m_start:M, 0:D-1, 0:D-1]
        + dx[:, m_start:M] * (1 - dy[:, m_start:M])     * corr[:, m_start:M, 0:D-1, 1:D  ]
        + (1 - dx[:, m_start:M]) * dy[:, m_start:M]     * corr[:, m_start:M, 1:D  , 0:D-1]
        + dx[:, m_start:M] * dy[:, m_start:M]           * corr[:, m_start:M, 1:D  , 1:D  ]
    )

    
    
    return out.permute(0, 1, 3, 2, 4, 5)



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
