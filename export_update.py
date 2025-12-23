#!/usr/bin/env python3
"""
Script to extract update model from DPVO and export to ONNX format with ONNX-compatible replacements.

Usage:
    python export_update.py --model dpvo.pth --output_dir ./onnx_models --max_edges 3000
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from dpvo.net import VONet, DIM
from dpvo.blocks import GradientClip, GatedResidual
from collections import OrderedDict


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

        # mask_ix = torch.ones_like(ix, dtype=net.dtype).view(1, -1, 1)  # use full mask
        # mask_jx = torch.ones_like(jx, dtype=net.dtype).view(1, -1, 1)

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


def load_dpvo_state_dict(model_path):
    """Load DPVO state dict and extract Update parameters"""
    print(f"📥 Loading {model_path} ...")
    dpvo_state = torch.load(model_path, map_location="cpu")
    print("✅ Model loaded successfully!")
    
    # Extract only Update parameters, ignoring lmbda.* keys
    update_state_dict = {
        k.replace("module.update.", ""): v
        for k, v in dpvo_state.items()
        if k.startswith("module.update.") and "lmbda" not in k
    }
    print(f"📝 Extracted {len(update_state_dict)} Update parameters")
    
    return update_state_dict


def export_to_onnx(model, dummy_inputs, output_path, input_names, output_names, opset_version=12, static_shape=True):
    """Export PyTorch model to ONNX format with static shapes for AMBA CV28"""
    print(f"\n🚀 Exporting to {output_path}...")
    print(f"  Input shapes:")
    for name, inp in zip(input_names, dummy_inputs):
        print(f"    {name}: {tuple(inp.shape)} dtype={inp.dtype}")
    print(f"  Output names: {output_names}")
    print(f"  Static shapes: {static_shape}")
    print(f"  Opset version: {opset_version}")

    # Export with static shapes (no dynamic axes) for AMBA CV28
    if static_shape:
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,  # ❌ Do NOT use dynamic axes for CV28
            opset_version=opset_version
        )
    else:
        # Dynamic shapes (for testing/debugging)
        dynamic_axes = {
            "net":  {2: "H"},
            "inp":  {2: "H"},
            "corr": {2: "H"},
            "ii": {1: "H"},
            "jj": {1: "H"},
            "kk": {1: "H"},
            "net_out": {2: "H"},
            "d_out":   {2: "H"},
            "w_out":   {2: "H"},
        }
        
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version
        )
    
    print(f"✅ Exported successfully to {output_path} 🎉")


def main():
    parser = argparse.ArgumentParser(description='Extract update model from DPVO and export to ONNX')
    parser.add_argument('--model', type=str, default='dpvo.pth',
                        help='Path to DPVO model file (default: dpvo.pth)')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                        help='Output directory for ONNX models (default: ./onnx_models)')
    parser.add_argument('--max_edges', type=int, default=756,
                        help='Maximum number of edges (static shape for AMBA CV28, default: 3000)')
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Patch size p (default: 3)')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version (default: 12)')
    parser.add_argument('--static', action='store_true', default=True,
                        help='Export with static shapes for AMBA CV28 (default: True)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------
    # 1️⃣ Load pretrained weights
    # ---------------------------
    update_state_dict = load_dpvo_state_dict(args.model)
    
    # ---------------------------
    # 2️⃣ Initialize UpdateONNX with matching p
    # ---------------------------
    DIM = 384
    p = args.patch_size
    print(f"\n{'='*60}")
    print("Model Information:")
    print(f"{'='*60}")
    print(f"Patch size (p): {p}")
    print(f"Update module DIM: {DIM}")
    print(f"Max edges (H): {args.max_edges}")
    print(f"Static shapes: {args.static} (for AMBA CV28)")
    
    update_onnx = UpdateONNX(p=p, export_onnx=True)
    update_onnx.load_state_dict(update_state_dict, strict=False)
    update_onnx.eval()
    print("🤖 UpdateONNX model initialized and ready")
    
    # ---------------------------
    # 3️⃣ Create dummy inputs (4D for CV28)
    # ---------------------------
    B, H, DIM = 1, args.max_edges, DIM
    
    net  = torch.randn(B, DIM, H, 1)
    inp  = torch.randn(B, DIM, H, 1)
    corr = torch.randn(B, 2*49*p*p, H, 1)
    
    # Create indices as [H, 1] first, then reshape to [1, H, 1]
    ii = torch.randint(0, 100, (H, 1), dtype=torch.int32)
    jj = torch.randint(0, 100, (H, 1), dtype=torch.int32)
    kk = torch.randint(0, 100, (H, 1), dtype=torch.int32)
    
    ii = ii.view(1, -1, 1)
    jj = jj.view(1, -1, 1)
    kk = kk.view(1, -1, 1)
    
    # ---------------------------
    # 4️⃣ Export to ONNX
    # ---------------------------
    update_path = output_dir / "update.onnx"
    
    dummy_inputs = (net, inp, corr, ii, jj, kk)
    input_names = ['net', 'inp', 'corr', 'ii', 'jj', 'kk']
    output_names = ['net_out', 'd_out', 'w_out']
    
    export_to_onnx(
        update_onnx,
        dummy_inputs,
        str(update_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        static_shape=args.static
    )
    
    print(f"\n{'='*60}")
    print("Export Summary:")
    print(f"{'='*60}")
    print(f"✅ Model exported to: {update_path}")
    print(f"📊 Input shapes:")
    print(f"    net: {tuple(net.shape)}")
    print(f"    inp: {tuple(inp.shape)}")
    print(f"    corr: {tuple(corr.shape)}")
    print(f"    ii: {tuple(ii.shape)} (int32)")
    print(f"    jj: {tuple(jj.shape)} (int32)")
    print(f"    kk: {tuple(kk.shape)} (int32)")
    print(f"📊 Expected output shapes:")
    print(f"    net_out: (1, {DIM}, {H}, 1)")
    print(f"    d_out: (1, 2, {H}, 1)")
    print(f"    w_out: (1, 2, {H}, 1)")
    print(f"\n{'='*60}")
    
    # ---------------------------
    # 4️⃣ Verify shapes
    # ---------------------------
    import onnx
    model = onnx.load(update_path)
    for inp in model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        print(f"Input '{inp.name}' shape: {shape}")
    for out in model.graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        print(f"Output '{out.name}' shape: {shape}")


if __name__ == '__main__':
    main()
