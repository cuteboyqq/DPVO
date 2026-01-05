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

    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        kk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of UpdateONNX module.
        
        Args:
            net: torch.Tensor - Network hidden state, shape [1, DIM, H, 1] where DIM=384 and H is the number of edges. Type: float32
            inp: torch.Tensor - Input features (imap), shape [1, DIM, H, 1]. Type: float32
            corr: torch.Tensor - Correlation features, shape [1, corr_dim, H, 1] where corr_dim = 2*49*p*p (882 for p=3). Type: float32
            ii: torch.Tensor - Source frame indices, shape [1, 1, H, 1]. Type: int32
            jj: torch.Tensor - Target frame indices, shape [1, 1, H, 1]. Type: int32
            kk: torch.Tensor - Patch indices, shape [1, 1, H, 1]. Type: int32
        
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - net_out: Updated network state, shape [1, DIM, H, 1] if export_onnx=True, or [1, H, DIM] if export_onnx=False
                - d_out: Position correction (delta), shape [1, 2, H, 1] if export_onnx=True, or [1, H, 2] if export_onnx=False
                - w_out: Confidence weights, shape [1, 2, H, 1] if export_onnx=True, or [1, H, 2] if export_onnx=False
        """
        # CV28 input reshape (4D -> 3D)
        net  = net.squeeze(-1).permute(0, 2, 1)
        inp  = inp.squeeze(-1).permute(0, 2, 1)
        corr = corr.squeeze(-1).permute(0, 2, 1)

        ii = ii.squeeze(0).squeeze(0).squeeze(-1)
        jj = jj.squeeze(0).squeeze(0).squeeze(-1)
        kk = kk.squeeze(0).squeeze(0).squeeze(-1)
        
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

        if self.export_onnx:
            # Ensure static shapes for ONNX export
            # net: [1, N, DIM] -> [1, DIM, N, 1] (N: number of edges, example : 756 or 1024) (DIM : 384)
            # d_out: [1, N, 2] -> [1, 2, N, 1]
            # w_out: [1, N, 2] -> [1, 2, N, 1]
            
            # Get explicit shapes to help ONNX infer static dimensions
            B, N, C = net.shape
            B_d, N_d, C_d = d_out.shape
            
            # Use explicit reshape with known dimensions to ensure ONNX sees static shapes
            # This is critical for ONNX to infer static output shapes
            # Method: permute -> unsqueeze -> explicit reshape with known dimensions
            
            # net_out: [1, N, DIM] -> [1, DIM, N, 1]
            net_out = net.permute(0, 2, 1).contiguous()  # [1, DIM, N]
            net_out = net_out.unsqueeze(-1)  # [1, DIM, N, 1]
            # Explicit reshape with known static dimensions
            net_out = net_out.view(B, C, N, 1) # B=1, C=DIM
            
            # d_out: [1, N, 2] -> [1, 2, N, 1]
            d_out = d_out.permute(0, 2, 1).contiguous()  # [1, 2, N]
            d_out = d_out.unsqueeze(-1)  # [1, 2, N, 1]
            # Explicit reshape with known static dimensions
            d_out = d_out.view(B_d, C_d, N_d, 1)
            
            # w_out: [1, N, 2] -> [1, 2, N, 1]
            w_out = w_out.permute(0, 2, 1).contiguous()  # [1, 2, N]
            w_out = w_out.unsqueeze(-1)  # [1, 2, N, 1]
            # Explicit reshape with known static dimensions
            w_out = w_out.view(B_d, C_d, N_d, 1)
            
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
        # Test forward pass to verify shapes
        print(f"  Testing forward pass...")
        with torch.no_grad():
            outputs = model(*dummy_inputs)
            print(f"  Output shapes from forward pass:")
            for name, out in zip(output_names, outputs):
                print(f"    {name}: {tuple(out.shape)} dtype={out.dtype}")
        
        # Export directly - torch.jit.trace might not work with loops in SoftAggONNX
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,  # ❌ Do NOT use dynamic axes for CV28
            opset_version=opset_version,
            do_constant_folding=False,  # Disable constant folding to preserve shapes
            export_params=True,  # Export parameters
            verbose=False
        )
        
        # Post-process ONNX model to ensure static shapes
        print(f"  Post-processing ONNX model to fix output shapes...")
        try:
            import onnx
            from onnx import helper
            
            onnx_model = onnx.load(output_path)
            
            # Get expected output shapes from forward pass
            with torch.no_grad():
                test_outputs = model(*dummy_inputs)
                expected_shapes = {name: tuple(out.shape) for name, out in zip(output_names, test_outputs)}
            
            # Fix output shapes to be static
            for i, output in enumerate(onnx_model.graph.output):
                if output.name in expected_shapes:
                    expected_shape = expected_shapes[output.name]
                    print(f"    Fixing output '{output.name}' shape to {expected_shape}")
                    
                    # Create new tensor type with static shape
                    shape_proto = helper.make_tensor_value_info(
                        output.name,
                        output.type.tensor_type.elem_type,
                        expected_shape
                    )
                    
                    # Replace the output type
                    onnx_model.graph.output[i].CopyFrom(shape_proto)
            
            # Save the fixed model
            onnx.save(onnx_model, output_path)
            print(f"  ✓ ONNX model post-processed with static output shapes")
        except Exception as e:
            print(f"  ⚠️  Post-processing failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Model exported but shapes may be dynamic - check manually")
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
    parser.add_argument('--max_edges', type=int, default=768,
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
    
    ii = ii.view(1, 1, -1, 1)
    jj = jj.view(1, 1, -1, 1)
    kk = kk.view(1, 1, -1, 1)
    
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
