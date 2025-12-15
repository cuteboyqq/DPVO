import torch
from torch_scatter import scatter_sum

from . import fastba
from . import lietorch
from .lietorch import SE3

from .utils import Timer

from . import projective_ops as pops

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def block_matmul(A, B):
    """ block matrix multiply """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

def block_solve(A, B, ep=1.0, lm=1e-4):
    """ block matrix solve """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)

    A = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)

    X = CholeskySolver.apply(A, B)
    return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A):
    import matplotlib.pyplot as plt
    b, n1, m1, p1, q1 = A.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()

'''
This BA function can run successfully
'''
def BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, ep=100.0, PRINT=False, fixedp=1, structure_only=False):
    """ bundle adjustment """

    b = 1
    n = max(ii.max().item(), jj.max().item()) + 1

    coords, v, (Ji, Jj, Jz) = \
        pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    p = coords.shape[3]
    r = targets - coords[...,p//2,p//2,:]

    v *= (r.norm(dim=-1) < 250).float()

    in_bounds = \
        (coords[...,p//2,p//2,0] > bounds[0]) & \
        (coords[...,p//2,p//2,1] > bounds[1]) & \
        (coords[...,p//2,p//2,0] < bounds[2]) & \
        (coords[...,p//2,p//2,1] < bounds[3])

    v *= in_bounds.float()

    if PRINT:
        print((r * v[...,None]).norm(dim=-1).mean().item())

    r = (v[...,None] * r).unsqueeze(dim=-1)    
    weights = (v[...,None] * weights).unsqueeze(dim=-1)

    wJiT = (weights * Ji).transpose(2,3) # Jacobian w.r.t. pose_i
    wJjT = (weights * Jj).transpose(2,3) # Jacobian w.r.t. pose_j
    wJzT = (weights * Jz).transpose(2,3) # Jacobian w.r.t. depth / structure

    Bii = torch.matmul(wJiT, Ji) # = J_i^T W J_i
    Bij = torch.matmul(wJiT, Jj) # = J_i^T W J_j
    Bji = torch.matmul(wJjT, Ji) # = J_j^T W J_i
    Bjj = torch.matmul(wJjT, Jj) # = J_j^T W J_j

    Eik = torch.matmul(wJiT, Jz)
    Ejk = torch.matmul(wJjT, Jz)

    vi = torch.matmul(wJiT, r)
    vj = torch.matmul(wJjT, r)

    # fix first pose
    ii = ii.clone()
    jj = jj.clone()

    n = n - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)

    B = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)

    E = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1) 

    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)

    v = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)

    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk, m)

    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C.shape)
        
    Q = 1.0 / (C + lmbda)
    
    ### solve w/ schur complement ###
    EQ = E * Q[:,None]

    if structure_only or n == 0:
        dZ = (Q * w).view(b, -1, 1, 1)

    else:
        S = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        dX = block_solve(S, y, ep=ep, lm=1e-4)

        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    x, y, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x, y, disps], dim=2)

    if not structure_only and n > 0:
        poses = pose_retr(poses, dX, fixedp + torch.arange(n))

    return poses, patches

def python_ba_wrapper(
    poses,
    patches,
    intrinsics,
    target,
    weight,
    lmbda,
    ii,
    jj,
    kk,
    PPF,          # unused (CUDA-specific)
    t0,
    t1,
    iterations,
    eff_impl=False  # ignored, Python BA is dense
):
    """
    Python replacement for CUDA fastba.BA
    Mirrors CUDA API and semantics.
    Updates poses and patches IN-PLACE.
    Returns None (like CUDA).
    """
    if isinstance(poses, torch.Tensor):
        poses = SE3(poses)
    else:
        poses = poses
    
    # -------------------------------------------------------
    # 1. Infer image bounds (CUDA does this internally)
    # -------------------------------------------------------
    # Assumption: intrinsics = [fx, fy, cx, cy]
    # Bounds are image-space pixel bounds
    #
    # You MUST adapt this if your image size comes elsewhere
    #
    device = poses.device
    dtype = poses.dtype

    # Common DPVO default (safe)
    H = int(2 * intrinsics[..., 3].max().item())
    W = int(2 * intrinsics[..., 2].max().item())

    # --- CUDA-style lambda expansion ---
    _, kk_inv = torch.unique(kk, return_inverse=True)
    m = int(kk_inv.max().item()) + 1

    if isinstance(lmbda, torch.Tensor) and lmbda.numel() == 1:
        lmbda = lmbda.view(1, 1, 1, 1).expand(1, m, 1, 1)
    
    bounds = torch.tensor(
        [0.0, 0.0, W - 1.0, H - 1.0],
        device=device,
        dtype=dtype
    )

    # -------------------------------------------------------
    # 2. Run BA for `iterations` (CUDA loop equivalent)
    # -------------------------------------------------------
    for _ in range(iterations):

        new_poses, new_patches = BA(
            poses=poses,
            patches=patches,
            intrinsics=intrinsics,
            targets=target,
            weights=weight,
            lmbda=lmbda,
            ii=ii,
            jj=jj,
            kk=kk,
            bounds=bounds,
            fixedp=t0,              # 🔑 exact CUDA mapping
            structure_only=(t1 - t0 == 0)
        )

        
        # -------------------------
        # Pose update (SE3-aware), where SE(3) = { R, t }, R ∈ SO(3) → 3D rotation, t ∈ ℝ³ → translation
        # -------------------------
        if isinstance(poses, SE3):
            poses = new_poses        # ✅ correct for SE3
        else:
            poses.copy_(new_poses)  # ✅ correct for Tensor

        
        # -------------------------
        # Patch update (Tensor)
        # -------------------------
        patches.copy_(new_patches)

    
    # 🔑 unwrap SE3 to tensor
    if hasattr(poses, "data"):
        poses = poses.data
    
    # -------------------------------------------------------
    # 4. CUDA BA returns nothing
    # -------------------------------------------------------
    return poses


'''
This code has error, l try to solve it , but still failed (20205-12-15 Alister updated)
'''
def BA_fast(
    poses,
    patches,
    intrinsics,
    targets,
    weights,
    lmbda,
    ii,
    jj,
    kk,
    bounds,
    ep=100.0,
    fixedp=1,
    structure_only=False
):
    """
    Fast bundle adjustment in PyTorch with minimal fusion.
    """

    b = 1  # batch size

    # ------------------------------------------------------------
    # 1. Transform patches
    # ------------------------------------------------------------
    coords, v, (Ji, Jj, Jz) = pops.transform(
        poses, patches, intrinsics, ii, jj, kk, jacobian=True
    )

    # Compute residuals
    p = coords.shape[3]
    r = targets - coords[..., p//2, p//2, :]

    # Apply threshold and bounds
    v *= (r.norm(dim=-1) < 250).float()
    in_bounds = (
        (coords[..., p//2, p//2, 0] > bounds[0]) &
        (coords[..., p//2, p//2, 1] > bounds[1]) &
        (coords[..., p//2, p//2, 0] < bounds[2]) &
        (coords[..., p//2, p//2, 1] < bounds[3])
    )
    v *= in_bounds.float()

    # Mask valid observations
    valid = v > 0
    if valid.sum() == 0:
        return poses, patches  # nothing to optimize

    # ------------------------------------------------------------
    # 2. Mask and flatten tensors
    # ------------------------------------------------------------
    # For each Jacobian and residual, keep only valid entries
    Ji = Ji[:, valid, :, :]      # [b, N_valid, patch_dim, 6]
    Jj = Jj[:, valid, :, :]
    Jz = Jz[:, valid, :, :]
    r  = r[:, valid, :][..., None]
    w  = weights[:, valid, :][..., None]

    ii = ii[valid]
    jj = jj[valid]
    kk = kk[valid]

    n = max(ii.max().item(), jj.max().item()) + 1 if len(ii) > 0 else 0

    # ------------------------------------------------------------
    # 3. Reindex structure points
    # ------------------------------------------------------------
    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)

    # ------------------------------------------------------------
    # 4. Flatten over patch dimension for einsum/scatter
    # ------------------------------------------------------------
    b, N, patch_dim = Ji.shape[:3]

    def flatten_patch_dim(x):
        return x.reshape(b * N * patch_dim, *x.shape[3:])

    Ji_flat = flatten_patch_dim(Ji)
    Jj_flat = flatten_patch_dim(Jj)
    Jz_flat = flatten_patch_dim(Jz)
    r_flat  = flatten_patch_dim(r)
    w_flat  = flatten_patch_dim(w)

    # Weighted Jacobians
    wJi_flat = w_flat * Ji_flat
    wJj_flat = w_flat * Jj_flat
    wJz_flat = w_flat * Jz_flat

    # ------------------------------------------------------------
    # 5. Compute block matrices
    # ------------------------------------------------------------
    Bii = wJi_flat.transpose(-1, -2) @ Ji_flat
    Bij = wJi_flat.transpose(-1, -2) @ Jj_flat
    Bji = wJj_flat.transpose(-1, -2) @ Ji_flat
    Bjj = wJj_flat.transpose(-1, -2) @ Jj_flat

    Eik = wJi_flat.transpose(-1, -2) @ Jz_flat
    Ejk = wJj_flat.transpose(-1, -2) @ Jz_flat

    vi = wJi_flat.transpose(-1, -2) @ r_flat
    vj = wJj_flat.transpose(-1, -2) @ r_flat
    w_struct = wJz_flat.transpose(-1, -2) @ r_flat
    C_struct = wJz_flat.transpose(-1, -2) @ Jz_flat

    # ------------------------------------------------------------
    # 6. Apply lambda / Q
    # ------------------------------------------------------------
    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C_struct.shape)
    Q = 1.0 / (C_struct + lmbda)

    # ------------------------------------------------------------
    # 7. Solve Schur complement
    # ------------------------------------------------------------
    EQ = Eik * Q[None]  # broadcasting over batch

    if structure_only or n == 0:
        dZ = (Q * w_struct).view(b, -1, 1, 1)
        dX = torch.zeros(b, n, 6, device=poses.device)
    else:
        # Build S = B - E Q E^T
        # Here we assume a helper function block_matmul is defined
        S = Bii + Bjj + Bji + Bij - block_matmul(EQ, Eik.transpose(-1, -2))
        y = vi + vj - block_matmul(EQ, w_struct)
        dX = block_solve(S, y, ep=ep, lm=1e-4)
        dZ = Q * (w_struct - block_matmul(Eik.transpose(-1, -2), dX).squeeze(-1))
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    # ------------------------------------------------------------
    # 8. Update patches & poses
    # ------------------------------------------------------------
    x, y, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x, y, disps], dim=2)

    if not structure_only and n > 0:
        poses = pose_retr(poses, dX, fixedp + torch.arange(n, device=poses.device))

    return poses, patches

