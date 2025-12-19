import torch
from torch_scatter import scatter_sum
from typing import Tuple, Union
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


def BA(
    poses: torch.Tensor,
    patches: torch.Tensor,
    intrinsics: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    lmbda: Union[float, torch.Tensor],
    ii: torch.Tensor,
    jj: torch.Tensor,
    kk: torch.Tensor,
    bounds: Tuple[float, float, float, float],
    ep: float = 100.0,
    PRINT: bool = False,
    fixedp: int = 1,
    structure_only: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bundle Adjustment (BA) solver used in DPVO.

    Solves the Schur-complement system:
        [ B  E ] [ dX ] = [ v ]
        [ Eᵀ C ] [ dZ ]   [ w ]

    Args:
        poses (torch.Tensor): Camera poses in se(3), shape (B, N, 6).
        patches (torch.Tensor): Patch state (x, y, inverse_depth), shape (B, M, 3).
        intrinsics (torch.Tensor): Camera intrinsics, shape (B, 4).
        targets (torch.Tensor): Target pixel coordinates, shape (B, M, 2).
        weights (torch.Tensor): Per-residual confidence weights, shape (B, M).
        lmbda (float | torch.Tensor): Levenberg–Marquardt damping factor.
        ii (torch.Tensor): Index of first pose per residual, shape (M,), dtype long.
        jj (torch.Tensor): Index of second pose per residual, shape (M,), dtype long.
        kk (torch.Tensor): Index of structure variable per residual, shape (M,), dtype long.
        bounds (Tuple[float, float, float, float]): Valid image bounds (xmin, ymin, xmax, ymax).
        ep (float, optional): Solver damping / convergence parameter, default 100.0.
        PRINT (bool, optional): Print mean reprojection error if True, default False.
        fixedp (int, optional): Number of initial poses to fix for gauge freedom, default 1.
        structure_only (bool, optional): Update only inverse depth if True, default False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            poses (torch.Tensor): Updated camera poses, shape (B, N, 6).
            patches (torch.Tensor): Updated patches with refined inverse depth, shape (B, M, 3).

    Notes:
        - Assumes batch size B = 1.
        - Structure Hessian is diagonal and inverted analytically.
        - Uses dynamic indexing and scatter-add operations.
        - Not suitable for static-shape accelerators (e.g., AMBA CV28).
    """

    # ---------------------------------------------------------
    # Basic setup
    # ---------------------------------------------------------
    b = 1  # DPVO uses batch size = 1
    n = max(ii.max().item(), jj.max().item()) + 1  # number of pose variables

    # ---------------------------------------------------------
    # Forward projection + Jacobians
    # ---------------------------------------------------------
    # coords : projected patch coordinates
    # v      : validity mask
    # Ji,Jj  : Jacobians wrt pose i and j (6 DoF)
    # Jz     : Jacobian wrt inverse depth
    coords, v, (Ji, Jj, Jz) = \
        pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    # ---------------------------------------------------------
    # Compute residual at patch center
    # ---------------------------------------------------------
    p = coords.shape[3]                      # patch size
    r = targets - coords[..., p//2, p//2, :] # reprojection residual

    # Reject large residuals
    v *= (r.norm(dim=-1) < 250).float()

    # Reject projections outside image bounds
    in_bounds = \
        (coords[..., p//2, p//2, 0] > bounds[0]) & \
        (coords[..., p//2, p//2, 1] > bounds[1]) & \
        (coords[..., p//2, p//2, 0] < bounds[2]) & \
        (coords[..., p//2, p//2, 1] < bounds[3])

    v *= in_bounds.float()

    if PRINT:
        print((r * v[..., None]).norm(dim=-1).mean().item())

    # Apply validity mask
    r = (v[..., None] * r).unsqueeze(dim=-1)
    weights = (v[..., None] * weights).unsqueeze(dim=-1)

    # ---------------------------------------------------------
    # Build weighted Jacobians
    # ---------------------------------------------------------
    wJiT = (weights * Ji).transpose(2, 3)
    wJjT = (weights * Jj).transpose(2, 3)
    wJzT = (weights * Jz).transpose(2, 3)

    # ---------------------------------------------------------
    # Pose–pose Hessian blocks
    # ---------------------------------------------------------
    Bii = torch.matmul(wJiT, Ji)
    Bij = torch.matmul(wJiT, Jj)
    Bji = torch.matmul(wJjT, Ji)
    Bjj = torch.matmul(wJjT, Jj)

    # ---------------------------------------------------------
    # Pose–structure Hessian blocks
    # ---------------------------------------------------------
    Eik = torch.matmul(wJiT, Jz)
    Ejk = torch.matmul(wJjT, Jz)

    # ---------------------------------------------------------
    # Gradient vectors
    # ---------------------------------------------------------
    vi = torch.matmul(wJiT, r)
    vj = torch.matmul(wJjT, r)

    # ---------------------------------------------------------
    # Fix first pose (gauge freedom)
    # ---------------------------------------------------------
    ii = ii.clone()
    jj = jj.clone()

    n = n - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    # ---------------------------------------------------------
    # Reindex structure variables
    # ---------------------------------------------------------
    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)  # number of structure variables

    # ---------------------------------------------------------
    # Assemble global pose Hessian B (block sparse)
    # ---------------------------------------------------------
    B = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)

    # ---------------------------------------------------------
    # Assemble pose–structure coupling matrix E
    # ---------------------------------------------------------
    E = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1)

    # ---------------------------------------------------------
    # Structure Hessian (diagonal)
    # ---------------------------------------------------------
    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)

    # ---------------------------------------------------------
    # Assemble gradient vectors
    # ---------------------------------------------------------
    v = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)

    w = safe_scatter_add_vec(torch.matmul(wJzT, r), kk, m)

    # ---------------------------------------------------------
    # Levenberg–Marquardt damping
    # ---------------------------------------------------------
    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C.shape)

    # Inverse of (C + λI) — diagonal, cheap
    Q = 1.0 / (C + lmbda)

    # ---------------------------------------------------------
    # Schur complement solve
    # ---------------------------------------------------------
    EQ = E * Q[:, None]  # E * C⁻¹

    if structure_only or n == 0:
        # Only update structure
        dZ = (Q * w).view(b, -1, 1, 1)

    else:
        # Pose-only system:
        # S = B − E C⁻¹ Eᵀ
        S = B - block_matmul(EQ, E.permute(0, 2, 1, 4, 3))

        # y = v − E C⁻¹ w
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))

        # Solve for pose increments
        dX = block_solve(S, y, ep=ep, lm=1e-4)

        # Back-substitute structure increments
        dZ = Q * (
            w - block_matmul(E.permute(0, 2, 1, 4, 3), dX).squeeze(dim=-1)
        )

        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    # ---------------------------------------------------------
    # Apply updates
    # ---------------------------------------------------------
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
