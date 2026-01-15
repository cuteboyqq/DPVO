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
import numpy as np

import torch.nn.functional as F

class UpdateONNX_CV28(nn.Module):
    def __init__(self, p, dim=384):
        super().__init__()
        self.dim = dim
        self.p = p

        # ------------------------
        # Correlation head
        # ------------------------
        corr_channels = 2 * 49 * p * p
        self.corr = nn.Sequential(
            nn.Conv2d(corr_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )

        # ------------------------
        # Neighbor transforms
        # ------------------------
        self.c1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )

        # ------------------------
        # Output heads
        # ------------------------
        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, 2, 1),
        )
        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, 2, 1),
            nn.Sigmoid(),
        )
        
        # create a dummy conv once
        self.keep_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.keep_conv.weight.data.zero_()
        self.keep_conv.requires_grad_(False)

    def neighbor_slice(self, net):
        """
        Fixed neighbor offsets using Slice + Pad (ONNX-safe)
        net: [B, C, H, 1]
        Returns:
            prev_net, next_net: [B, C, H, 1]
        """
        # Previous neighbor: shift right by 1, pad left
        prev_net = F.pad(net[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')

        # Next neighbor: shift left by 1, pad right
        next_net = F.pad(net[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')

        return prev_net, next_net

    def forward(self, net, inp, corr, ii, jj, kk):
        """
        net, inp: [B, C, H, 1]
        corr    : [B, CorrC, H, 1]
        ii,jj,kk: [B, 1, H, 1] int32 (kept as ONNX inputs)
        """
        dummy_ii = self.keep_conv(ii)
        dummy_jj = self.keep_conv(jj)
        dummy_kk = self.keep_conv(kk)

        net = net + (dummy_ii + dummy_jj + dummy_kk) * 0.0
        
        # 1️⃣ Correlation
        net = net + inp + self.corr(corr)

        # 2️⃣ Neighbor slices (static ±1 neighbors)
        prev_net, next_net = self.neighbor_slice(net)
        net = net + self.c1(prev_net) + self.c2(next_net)

        # 3️⃣ Outputs
        d_out = self.d(net)
        w_out = self.w(net)

        # ✅ Keep ii/jj/kk as inputs in ONNX, even if unused
        # _ = ii + jj + kk  # dummy op to register inputs

        return net, d_out, w_out




class SoftAggReduceMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Conv2d(dim, dim, 1)
        self.g = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # x: [1, C, H, 1]

        fx = self.f(x)
        gx = self.g(x)

        # channel-wise gating (YOLO style)
        w = torch.sigmoid(gx)
        return fx * w



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

def export_to_onnx(
    model,
    dummy_inputs,
    output_path,
    input_names,
    output_names,
    opset_version=12,
    static_shape=True,
):
    """Export PyTorch model to ONNX format with static shapes for AMBA CV28"""

    import torch

    print(f"\n🚀 Exporting to {output_path}...")
    print(f"  Input shapes:")
    for name, inp in zip(input_names, dummy_inputs):
        print(f"    {name}: {tuple(inp.shape)} dtype={inp.dtype}")
    print(f"  Output names: {output_names}")
    print(f"  Static shapes: {static_shape}")
    print(f"  Opset version: {opset_version}")

    # ------------------------------------------------------------------
    # Helper: collect intermediate tensor shapes from PyTorch
    # ------------------------------------------------------------------
    def collect_intermediate_shapes(model, dummy_inputs):
        """
        Run a forward pass and collect output shapes of leaf modules.
        Returns: dict {module_name: shape}
        """
        shape_dict = {}
        hooks = []

        def hook_fn(name):
            def hook(module, inputs, output):
                if torch.is_tensor(output):
                    shape_dict[name] = tuple(output.shape)
                elif isinstance(output, (list, tuple)):
                    for o in output:
                        if torch.is_tensor(o):
                            shape_dict[name] = tuple(o.shape)
                            break
            return hook

        for name, module in model.named_modules():
            # only leaf modules
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            model(*dummy_inputs)

        for h in hooks:
            h.remove()

        return shape_dict

    # ------------------------------------------------------------------
    # Helper: find matching PyTorch shape for an ONNX value name
    # ------------------------------------------------------------------
    def find_matching_shape(onnx_name, shape_dict):
        """
        Heuristic mapping:
        PyTorch module name: update.net.0
        ONNX value name:     /update/net/Conv_output_0
        """
        for k, shape in shape_dict.items():
            if k.replace(".", "/") in onnx_name:
                return shape
        return None

    # ------------------------------------------------------------------
    # Export logic
    # ------------------------------------------------------------------
    if static_shape:
        # --------------------------------------------------------------
        # 1. Test forward pass (sanity + output shapes)
        # --------------------------------------------------------------
        print(f"  Testing forward pass...")
        with torch.no_grad():
            outputs = model(*dummy_inputs)
            print(f"  Output shapes from forward pass:")
            for name, out in zip(output_names, outputs):
                print(f"    {name}: {tuple(out.shape)} dtype={out.dtype}")

        # --------------------------------------------------------------
        # 2. Export ONNX (NO dynamic axes)
        # --------------------------------------------------------------
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,          # ❌ CV28 does not allow dynamic axes
            opset_version=opset_version,
            do_constant_folding=False,  # preserve shape semantics
            export_params=True,
            verbose=False,
        )

        # --------------------------------------------------------------
        # 3. Post-process ONNX: fix outputs + intermediates
        # --------------------------------------------------------------
        print(f"  Post-processing ONNX model to fix output & intermediate shapes...")
        print(f"  (Does NOT change computation, only ONNX shape metadata)")

        try:
            import onnx
            from onnx import helper

            onnx_model = onnx.load(output_path)

            # ----------------------------------------------------------
            # 3a. Fix OUTPUT shapes
            # ----------------------------------------------------------
            with torch.no_grad():
                test_outputs = model(*dummy_inputs)
                expected_output_shapes = {
                    name: tuple(out.shape)
                    for name, out in zip(output_names, test_outputs)
                }

            for i, output in enumerate(onnx_model.graph.output):
                if output.name in expected_output_shapes:
                    shape = expected_output_shapes[output.name]
                    print(f"    Fixing output '{output.name}' → {shape}")

                    new_vi = helper.make_tensor_value_info(
                        output.name,
                        output.type.tensor_type.elem_type,
                        shape,
                    )
                    onnx_model.graph.output[i].CopyFrom(new_vi)

            # ----------------------------------------------------------
            # 3b. Fix INTERMEDIATE tensor shapes (graph.value_info)
            # ----------------------------------------------------------
            print(f"  Collecting PyTorch intermediate tensor shapes...")
            intermediate_shapes = collect_intermediate_shapes(model, dummy_inputs)

            fixed_cnt = 0
            for i, value_info in enumerate(onnx_model.graph.value_info):
                # Skip if shape is already fully static
                dims = value_info.type.tensor_type.shape.dim
                if dims and all(d.HasField("dim_value") for d in dims):
                    continue

                shape = find_matching_shape(value_info.name, intermediate_shapes)
                if shape is None:
                    continue

                print(f"    Fixing intermediate '{value_info.name}' → {shape}")

                new_vi = helper.make_tensor_value_info(
                    value_info.name,
                    value_info.type.tensor_type.elem_type,
                    shape,
                )
                onnx_model.graph.value_info[i].CopyFrom(new_vi)
                fixed_cnt += 1

            print(f"  ✓ Fixed {fixed_cnt} intermediate tensors")

            # ----------------------------------------------------------
            # 3c. Save patched ONNX
            # ----------------------------------------------------------
            onnx.save(onnx_model, output_path)

            print(f"  ✓ ONNX model fully patched for CV28")
            print(f"    - outputs: static")
            print(f"    - intermediates: static")
            print(f"    - shape propagation: deterministic")

        except Exception as e:
            print(f"  ⚠️ Post-processing failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Model exported, but shapes may still be symbolic")

    else:
        # --------------------------------------------------------------
        # Dynamic export (debug only)
        # --------------------------------------------------------------
        dynamic_axes = {
            "net":  {2: "H"},
            "inp":  {2: "H"},
            "corr": {2: "H"},
            "ii":   {1: "H"},
            "jj":   {1: "H"},
            "kk":   {1: "H"},
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
            opset_version=opset_version,
        )

    print(f"✅ Exported successfully to {output_path} 🎉")
    

def load_linear_as_conv1x1(model, state_dict):
    """
    Load Linear weights into Conv1x1 layers by reshaping:
    [out, in] -> [out, in, 1, 1]
    """
    new_state = {}

    for k, v in state_dict.items():
        if k not in model.state_dict():
            continue

        target = model.state_dict()[k]

        # Linear -> Conv1x1 weight
        if v.ndim == 2 and target.ndim == 4:
            print(f"🔁 Reshaping {k}: {tuple(v.shape)} → {tuple(target.shape)}")
            new_state[k] = v.unsqueeze(-1).unsqueeze(-1)

        # Bias or already matching
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)


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
    
    update_onnx = UpdateONNX_CV28(p=3)
    # update_onnx = UpdateCV28Minimal(dim=384, export_onnx=True)
    # update_onnx.load_state_dict(update_state_dict, strict=False)
    # load_linear_as_conv1x1(update_onnx, update_state_dict)
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
    # Create indices directly as float32
    ii = torch.randint(0, 100, (H, 1), dtype=torch.int32).float()
    jj = torch.randint(0, 100, (H, 1), dtype=torch.int32).float()
    kk = torch.randint(0, 100, (H, 1), dtype=torch.int32).float()

    
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
    print(f"    ii: {tuple(ii.shape)} (float)")
    print(f"    jj: {tuple(jj.shape)} (float)")
    print(f"    kk: {tuple(kk.shape)} (float)")
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
