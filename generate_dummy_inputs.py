#!/usr/bin/env python3
"""
Generate dummy input .bin files for AMBA CV28 model conversion.

These dummy inputs match the 4D static shapes exported in update.onnx:
- net:  [1, 1, N, 384]
- inp:  [1, 1, N, 384]
- corr: [1, 1, N, 882]
- flow: [1, 1, N, 2]
- ii:   [1, 1, N, 1] (int32 - AMBA CV28 converter expects int32)
- jj:   [1, 1, N, 1] (int32 - AMBA CV28 converter expects int32)
- kk:   [1, 1, N, 1] (int32 - AMBA CV28 converter expects int32)

Usage:
    python generate_dummy_inputs.py --max_edges 3000 --output_dir ./dummy
"""

import numpy as np
import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Generate dummy input .bin files for AMBA CV28')
    parser.add_argument('--max_edges', type=int, default=3000,
                        help='Maximum number of edges (default: 3000)')
    parser.add_argument('--output_dir', type=str, default='./dummy',
                        help='Output directory for dummy files (default: ./dummy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Configuration
    N = args.max_edges
    DIM = 384
    corr_dim = 2 * 49 * 3 * 3  # 2 pyramid levels * 49 (7x7) * 3 * 3 patches = 882
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each input
    net_out_dir = output_dir / "dummy_net"
    inp_out_dir = output_dir / "dummy_inp"
    corr_out_dir = output_dir / "dummy_corr"
    flow_out_dir = output_dir / "dummy_flow"
    ii_out_dir = output_dir / "dummy_ii"
    jj_out_dir = output_dir / "dummy_jj"
    kk_out_dir = output_dir / "dummy_kk"
    
    for dir_path in [net_out_dir, inp_out_dir, corr_out_dir, flow_out_dir, 
                     ii_out_dir, jj_out_dir, kk_out_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print("Generating dummy input .bin files for AMBA CV28")
    print(f"{'='*60}")
    print(f"Max edges (N): {N}")
    print(f"DIM: {DIM}")
    print(f"Corr dim: {corr_dim}")
    print(f"Output directory: {output_dir}")
    print()
    
    # ---------------------------
    # Generate dummy inputs with 4D shapes
    # ---------------------------
    
    # net: [1, 1, N, DIM] - Network hidden state
    net = np.random.randn(1, 1, N, DIM).astype(np.float32)
    net_path = net_out_dir / "net.bin"
    net.tofile(str(net_path))
    print(f"✓ Generated net.bin: {net.shape} -> {net_path}")
    
    # inp: [1, 1, N, DIM] - Input features (imap)
    inp = np.random.randn(1, 1, N, DIM).astype(np.float32)
    inp_path = inp_out_dir / "inp.bin"
    inp.tofile(str(inp_path))
    print(f"✓ Generated inp.bin: {inp.shape} -> {inp_path}")
    
    # corr: [1, 1, N, corr_dim] - Correlation features
    corr = np.random.randn(1, 1, N, corr_dim).astype(np.float32)
    corr_path = corr_out_dir / "corr.bin"
    corr.tofile(str(corr_path))
    print(f"✓ Generated corr.bin: {corr.shape} -> {corr_path}")
    
    # flow: [1, 1, N, 2] - Dummy flow tensor (not used, can be zeros)
    flow = np.zeros((1, 1, N, 2), dtype=np.float32)
    flow_path = flow_out_dir / "flow.bin"
    flow.tofile(str(flow_path))
    print(f"✓ Generated flow.bin: {flow.shape} -> {flow_path}")
    
    # ii: [1, 1, N, 1] - Source frame indices
    # Note: AMBA CV28 converter may expect int32 instead of int64
    # Generate indices in valid range (0 to 35 for 36 frames)
    # Try int32 first (if converter fails, may need to check converter config)
    ii = np.random.randint(0, 36, size=(1, 1, N, 1), dtype=np.int32)
    ii_path = ii_out_dir / "ii.bin"
    ii.tofile(str(ii_path))
    print(f"✓ Generated ii.bin: {ii.shape} (int32) -> {ii_path}")
    
    # jj: [1, 1, N, 1] - Target frame indices
    jj = np.random.randint(0, 36, size=(1, 1, N, 1), dtype=np.int32)
    jj_path = jj_out_dir / "jj.bin"
    jj.tofile(str(jj_path))
    print(f"✓ Generated jj.bin: {jj.shape} (int32) -> {jj_path}")
    
    # kk: [1, 1, N, 1] - Patch indices
    # Generate indices in valid range (0 to N-1)
    kk = np.random.randint(0, N, size=(1, 1, N, 1), dtype=np.int32)
    kk_path = kk_out_dir / "kk.bin"
    kk.tofile(str(kk_path))
    print(f"✓ Generated kk.bin: {kk.shape} (int32) -> {kk_path}")
    
    # ---------------------------
    # Summary
    # ---------------------------
    print()
    print(f"{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"All dummy .bin files saved in: {output_dir}")
    print()
    print("Input shapes (4D for AMBA CV28):")
    print(f"  net:  {net.shape}  -> {net_path}")
    print(f"  inp:  {inp.shape}  -> {inp_path}")
    print(f"  corr: {corr.shape} -> {corr_path}")
    print(f"  flow: {flow.shape} -> {flow_path}")
    print(f"  ii:   {ii.shape}   -> {ii_path} (int32)")
    print(f"  jj:   {jj.shape}   -> {jj_path} (int32)")
    print(f"  kk:   {kk.shape}   -> {kk_path} (int32)")
    print()
    print("File sizes:")
    print(f"  net.bin:  {net_path.stat().st_size / 1024:.2f} KB")
    print(f"  inp.bin:  {inp_path.stat().st_size / 1024:.2f} KB")
    print(f"  corr.bin: {corr_path.stat().st_size / 1024:.2f} KB")
    print(f"  flow.bin: {flow_path.stat().st_size / 1024:.2f} KB")
    print(f"  ii.bin:   {ii_path.stat().st_size / 1024:.2f} KB")
    print(f"  jj.bin:   {jj_path.stat().st_size / 1024:.2f} KB")
    print(f"  kk.bin:   {kk_path.stat().st_size / 1024:.2f} KB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()


# Command: Generate dummy inputs for AMBA CV28
# python generate_dummy_inputs.py --max_edges 3000 --output_dir ./dummy
