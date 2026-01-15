#!/usr/bin/env python3
"""
Script to extract fnet and inet from DPVO model and export to ONNX format.

Usage:
    python export_fnet_inet.py --model dpvo.pth --output_dir ./onnx_models
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from dpvo.net import VONet
from collections import OrderedDict


class FNetWrapper(nn.Module):
    """Wrapper for fnet to handle input normalization and static 4D input for AMBA CV28"""
    def __init__(self, fnet, num_frames=36):
        super(FNetWrapper, self).__init__()
        self.fnet = fnet
        self.num_frames = num_frames
    
    def forward(self, images):
        """
        Args:
            images: [N, 3, H, W] - Input images (4D, static shape for AMBA CV28)
                   N = num_frames (36), normalized to [-0.5, 0.5]
        Returns:
            fmap: [N, 128, H/4, W/4] - Feature map (4D, divided by 4.0)
        """
        # Reshape from 4D [N, 3, H, W] to 5D [1, N, 3, H, W] for internal processing
        b, c, h, w = images.shape
        images_5d = images.unsqueeze(0)  # [1, N, 3, H, W]
        
        # Process through fnet
        fmap_5d = self.fnet(images_5d) / 4.0  # [1, N, 128, H/4, W/4]
        
        # Reshape back to 4D [N, 128, H/4, W/4]
        fmap = fmap_5d.squeeze(0)  # [N, 128, H/4, W/4]
        return fmap


class INetWrapper(nn.Module):
    """Wrapper for inet to handle input normalization and static 4D input for AMBA CV28"""
    def __init__(self, inet, num_frames=36):
        super(INetWrapper, self).__init__()
        self.inet = inet
        self.num_frames = num_frames
    
    def forward(self, images):
        """
        Args:
            images: [N, 3, H, W] - Input images (4D, static shape for AMBA CV28)
                   N = num_frames (36), normalized to [-0.5, 0.5]
        Returns:
            imap: [N, 384, H/4, W/4] - Feature map (4D, divided by 4.0)
        """
        # Reshape from 4D [N, 3, H, W] to 5D [1, N, 3, H, W] for internal processing
        b, c, h, w = images.shape
        images_5d = images.unsqueeze(0)  # [1, N, 3, H, W]
        
        # Process through inet
        imap_5d = self.inet(images_5d) / 4.0  # [1, N, 384, H/4, W/4]
        
        # Reshape back to 4D [N, 384, H/4, W/4]
        imap = imap_5d.squeeze(0)  # [N, 384, H/4, W/4]
        return imap


def load_dpvo_model(model_path):
    """Load DPVO model from checkpoint"""
    print(f"📥 Loading model from {model_path}...")
    
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        if "update.lmbda" not in k:
            new_state_dict[k.replace('module.', '')] = v
    
    network = VONet()
    network.load_state_dict(new_state_dict)
    network.eval()
    
    print("✅ Model loaded successfully!")
    return network


def export_to_onnx(model, input_shape, output_path, input_names, output_names, opset_version=11, static_shape=True):
    """Export PyTorch model to ONNX format with static shapes for AMBA CV28"""
    print(f"\n🚀 Exporting to {output_path}...")
    print(f"  📊 Input shape: {input_shape} (static: {static_shape})")
    print(f"  📥 Input names: {input_names}")
    print(f"  📤 Output names: {output_names}")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export with static shapes (no dynamic axes) for AMBA CV28
    if static_shape:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            # No dynamic_axes - all dimensions are static
            verbose=False
        )
    else:
        # Dynamic shapes (original behavior)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: 'batch', 1: 'num_frames'},  # Allow dynamic batch and frame count
            } if len(input_shape) == 5 else {
                input_names[0]: {0: 'batch'},  # Allow dynamic batch
            },
            verbose=False
        )
    
    print(f"  ✅ Exported successfully to {output_path} 🎉")


def main():
    parser = argparse.ArgumentParser(description='Extract fnet and inet from DPVO model')
    parser.add_argument('--model', type=str, default='dpvo.pth',
                        help='Path to DPVO model file (default: dpvo.pth)')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                        help='Output directory for ONNX models (default: ./onnx_models)')
    parser.add_argument('--height', type=int, default=528,
                        help='Input image height (default: 528)')
    parser.add_argument('--width', type=int, default=960,
                        help='Input image width (default: 960)')
    parser.add_argument('--num_frames', type=int, default=1,
                        help='Number of frames in input (default: 36 for AMBA CV28)')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--static', action='store_true', default=True,
                        help='Export with static shapes for AMBA CV28 (default: True)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_dpvo_model(args.model)
    
    # Extract fnet and inet
    fnet = model.patchify.fnet
    inet = model.patchify.inet
    
    print(f"\n{'='*60}")
    print("📋 Model Information:")
    print(f"{'='*60}")
    print(f"  🔹 FNet output_dim: 128")
    print(f"  🔹 INet output_dim: 384")
    print(f"  📐 Input resolution: {args.height} x {args.width}")
    print(f"  📐 Output resolution (after /4): {args.height//4} x {args.width//4}")
    print(f"  🎬 Number of frames: {args.num_frames}")
    print(f"  🔒 Static shapes: {args.static} (for AMBA CV28)")
    
    # Wrap models to include normalization
    # For AMBA CV28: use 4D input [N, 3, H, W] instead of 5D [B, N, 3, H, W]
    print(f"\n🔧 Creating ONNX-compatible wrappers...")
    fnet_wrapped = FNetWrapper(fnet, num_frames=args.num_frames)
    inet_wrapped = INetWrapper(inet, num_frames=args.num_frames)
    
    # Set to eval mode
    fnet_wrapped.eval()
    inet_wrapped.eval()
    print(f"  ✅ Wrappers created successfully")
    
    if args.static:
        # Static 4D input for AMBA CV28: [N, 3, H, W]
        # N = num_frames (36), batch dimension removed
        input_shape = (args.num_frames, 3, args.height, args.width)
        output_fmap_shape = (args.num_frames, 128, args.height//4, args.width//4)
        output_imap_shape = (args.num_frames, 384, args.height//4, args.width//4)
    else:
        # Original 5D input: [B, N, 3, H, W]
        input_shape = (1, args.num_frames, 3, args.height, args.width)
        output_fmap_shape = (1, args.num_frames, 128, args.height//4, args.width//4)
        output_imap_shape = (1, args.num_frames, 384, args.height//4, args.width//4)
    
    # Export fnet
    print(f"\n📤 Exporting FNet model...")
    fnet_path = output_dir / "fnet.onnx"
    export_to_onnx(
        fnet_wrapped,
        input_shape,
        str(fnet_path),
        input_names=['images'],
        output_names=['fmap'],
        opset_version=args.opset,
        static_shape=args.static
    )
    
    # Export inet
    print(f"\n📤 Exporting INet model...")
    inet_path = output_dir / "inet.onnx"
    export_to_onnx(
        inet_wrapped,
        input_shape,
        str(inet_path),
        input_names=['images'],
        output_names=['imap'],
        opset_version=args.opset,
        static_shape=args.static
    )
    
    print(f"\n{'='*60}")
    print("📊 Export Summary:")
    print(f"{'='*60}")
    print(f"  ✅ FNet exported to: {fnet_path}")
    print(f"  ✅ INet exported to: {inet_path}")
    print(f"\n📥 Input format:")
    if args.static:
        print(f"  📐 Shape: [N, 3, H, W] = [{args.num_frames}, 3, {args.height}, {args.width}] (4D, static)")
    else:
        print(f"  📐 Shape: [B, N, 3, H, W] = [1, {args.num_frames}, 3, {args.height}, {args.width}] (5D)")
    print(f"  🔢 Range: [-0.5, 0.5] (normalized from [0, 255])")
    print(f"  📝 Formula: images = 2 * (images / 255.0) - 0.5")
    print(f"\n📤 Output format:")
    if args.static:
        print(f"  🔹 FNet: [N, 128, H/4, W/4] = [{args.num_frames}, 128, {args.height//4}, {args.width//4}] (4D, static)")
        print(f"  🔹 INet: [N, 384, H/4, W/4] = [{args.num_frames}, 384, {args.height//4}, {args.width//4}] (4D, static)")
    else:
        print(f"  🔹 FNet: [B, N, 128, H/4, W/4] = [1, {args.num_frames}, 128, {args.height//4}, {args.width//4}] (5D)")
        print(f"  🔹 INet: [B, N, 384, H/4, W/4] = [1, {args.num_frames}, 384, {args.height//4}, {args.width//4}] (5D)")
    print(f"  ⚠️  Both outputs are already divided by 4.0")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()


# Command : # Export with static 4D shapes for AMBA CV28 (default)
# python export_fnet_inet.py --model dpvo.pth --output_dir ./onnx_models --num_frames 36 --height 480 --width 640

# Or with custom resolution
# python export_fnet_inet.py --model dpvo.pth --output_dir ./onnx_models --num_frames 36 --height 480 --width 640 --static
