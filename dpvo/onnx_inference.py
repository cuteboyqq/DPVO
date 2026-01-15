"""
ONNX Inference Wrapper for DPVO Models

This module provides ONNX Runtime inference for fnet, inet, and update models,
replacing PyTorch model inference.
"""

import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Tuple


class ONNXInference:
    """ONNX Runtime inference wrapper for DPVO models"""
    
    def __init__(self, fnet_path: Optional[str] = None, 
                 inet_path: Optional[str] = None,
                 update_path: Optional[str] = None,
                 providers: Optional[list] = None):
        """
        Initialize ONNX inference sessions.
        
        Args:
            fnet_path: Path to fnet.onnx model
            inet_path: Path to inet.onnx model
            update_path: Path to update.onnx model
            providers: ONNX Runtime execution providers (default: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.fnet_session = None
        self.inet_session = None
        self.update_session = None
        self.max_edges = None  # Will be set when update model is loaded
        
        if fnet_path:
            self.fnet_session = self._create_session(fnet_path, providers)
        if inet_path:
            self.inet_session = self._create_session(inet_path, providers)
        if update_path:
            self.update_session = self._create_session(update_path, providers)
    
    def _create_session(self, model_path: str, providers: list) -> ort.InferenceSession:
        """Create ONNX Runtime inference session"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        print(f"Loaded ONNX model: {model_path.name}")
        print(f"  Inputs: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Outputs: {[out.name for out in session.get_outputs()]}")
        
        # Extract MAX_EDGE_NUM from update model if this is the update session
        if model_path.name == "update.onnx":
            # Get the shape of ii input (should be [1, 1, MAX_EDGE_NUM, 1])
            ii_input = session.get_inputs()[3]  # ii is the 4th input (index 3)
            if len(ii_input.shape) == 4 and ii_input.shape[2] > 0:
                self.max_edges = int(ii_input.shape[2])
                print(f"  Detected MAX_EDGE_NUM: {self.max_edges}")
            else:
                self.max_edges = None
        
        return session
    
    def fnet_forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run fnet inference.
        
        Args:
            images: Input images, shape [B, N, 3, H, W] (typically [1, 1, 3, H, W])
                   Expected to be normalized to [-0.5, 0.5]
        
        Returns:
            fmap: Feature map, shape [B, N, 128, H/4, W/4] (typically [1, 1, 128, H/4, W/4])
        """
        if self.fnet_session is None:
            raise RuntimeError("fnet ONNX model not loaded")
        
        # Handle input shape conversion
        # ONNX models expect [N, 3, H, W] but DPVO provides [B, N, 3, H, W]
        original_shape = images.shape
        if len(original_shape) == 5:
            # [B, N, 3, H, W] -> [N, 3, H, W] (squeeze batch)
            # For DPVO, B=1, N=1 typically, so we get [1, 3, H, W]
            images = images.squeeze(0)  # [1, 3, H, W]
        
        # Convert to numpy and ensure correct dtype
        # Note: ONNX models expect inputs on CPU, but we'll move outputs back to GPU
        if images.is_cuda:
            images_np = images.cpu().numpy().astype(np.float32)
        else:
            images_np = images.numpy().astype(np.float32)
        
        # Run inference
        input_name = self.fnet_session.get_inputs()[0].name
        output_name = self.fnet_session.get_outputs()[0].name
        outputs = self.fnet_session.run([output_name], {input_name: images_np})
        
        # Convert back to torch tensor and move to GPU
        fmap = torch.from_numpy(outputs[0])  # [N, 128, H/4, W/4] or [1, 128, H/4, W/4]
        if torch.cuda.is_available():
            fmap = fmap.cuda()
        
        # Reshape back to original batch structure if needed
        if len(original_shape) == 5:
            fmap = fmap.unsqueeze(0)  # [1, 128, H/4, W/4] -> [1, 1, 128, H/4, W/4]
        
        return fmap
    
    def inet_forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run inet inference.
        
        Args:
            images: Input images, shape [B, N, 3, H, W] (typically [1, 1, 3, H, W])
                   Expected to be normalized to [-0.5, 0.5]
        
        Returns:
            imap: Feature map, shape [B, N, 384, H/4, W/4] (typically [1, 1, 384, H/4, W/4])
        """
        if self.inet_session is None:
            raise RuntimeError("inet ONNX model not loaded")
        
        # Handle input shape conversion
        # ONNX models expect [N, 3, H, W] but DPVO provides [B, N, 3, H, W]
        original_shape = images.shape
        if len(original_shape) == 5:
            # [B, N, 3, H, W] -> [N, 3, H, W] (squeeze batch)
            # For DPVO, B=1, N=1 typically, so we get [1, 3, H, W]
            images = images.squeeze(0)  # [1, 3, H, W]
        
        # Convert to numpy and ensure correct dtype
        # Note: ONNX models expect inputs on CPU, but we'll move outputs back to GPU
        if images.is_cuda:
            images_np = images.cpu().numpy().astype(np.float32)
        else:
            images_np = images.numpy().astype(np.float32)
        
        # Run inference
        input_name = self.inet_session.get_inputs()[0].name
        output_name = self.inet_session.get_outputs()[0].name
        outputs = self.inet_session.run([output_name], {input_name: images_np})
        
        # Convert back to torch tensor and move to GPU
        imap = torch.from_numpy(outputs[0])  # [N, 384, H/4, W/4] or [1, 384, H/4, W/4]
        if torch.cuda.is_available():
            imap = imap.cuda()
        
        # Reshape back to original batch structure if needed
        if len(original_shape) == 5:
            imap = imap.unsqueeze(0)  # [1, 384, H/4, W/4] -> [1, 1, 384, H/4, W/4]
        
        return imap
    
    def update_forward(self, net: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor,
                    ii: torch.Tensor, jj: torch.Tensor, kk: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        """
        Run update model inference.
        
        Args:
            net: Network hidden state, shape [1, H, DIM] where H is variable
            inp: Input features (imap), shape [1, H, DIM]
            corr: Correlation features, shape [1, H, corr_dim]
            ii: Source frame indices, shape [H] or [1, H]
            jj: Target frame indices, shape [H] or [1, H]
            kk: Patch indices, shape [H] or [1, H]
        
        Returns:
            net: Updated network state, shape [1, H, DIM] (original H, not padded)
            (delta, weight, None): Tuple containing:
                - delta: Position correction, shape [1, H, 2]
                - weight: Confidence weights, shape [1, H, 2]
                - None: Placeholder
        """
        if self.update_session is None:
            raise RuntimeError("update ONNX model not loaded")
        
        if self.max_edges is None:
            raise RuntimeError("MAX_EDGE_NUM not detected from ONNX model")
        
        # Get input/output names
        input_names = [inp.name for inp in self.update_session.get_inputs()]
        output_names = [out.name for out in self.update_session.get_outputs()]
        
        # Get original H (number of edges)
        if len(net.shape) == 3:
            original_H = net.shape[1]  # [1, H, DIM]
        else:
            raise ValueError(f"Unexpected net shape: {net.shape}, expected [1, H, DIM]")
        
        # Prepare inputs - convert from [1, H, DIM] to [1, DIM, H, 1] for CV28
        net_4d = net.permute(0, 2, 1).unsqueeze(-1)  # [1, H, DIM] -> [1, DIM, H, 1]
        inp_4d = inp.permute(0, 2, 1).unsqueeze(-1)  # [1, H, DIM] -> [1, DIM, H, 1]
        corr_4d = corr.permute(0, 2, 1).unsqueeze(-1)  # [1, H, corr_dim] -> [1, corr_dim, H, 1]
        
        # Handle index tensors - convert to [1, 1, H, 1] format
        if len(ii.shape) == 1:
            ii_4d = ii.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [H] -> [1, 1, H, 1]
        elif len(ii.shape) == 2:
            ii_4d = ii.unsqueeze(0).unsqueeze(-1)  # [1, H] -> [1, 1, H, 1]
        else:
            ii_4d = ii  # Already in correct shape
        
        if len(jj.shape) == 1:
            jj_4d = jj.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [H] -> [1, 1, H, 1]
        elif len(jj.shape) == 2:
            jj_4d = jj.unsqueeze(0).unsqueeze(-1)  # [1, H] -> [1, 1, H, 1]
        else:
            jj_4d = jj  # Already in correct shape
        
        if len(kk.shape) == 1:
            kk_4d = kk.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [H] -> [1, 1, H, 1]
        elif len(kk.shape) == 2:
            kk_4d = kk.unsqueeze(0).unsqueeze(-1)  # [1, H] -> [1, 1, H, 1]
        else:
            kk_4d = kk  # Already in correct shape
        
        # Pad or slice to MAX_EDGE_NUM
        MAX_EDGE_NUM = self.max_edges
        if original_H < MAX_EDGE_NUM:
            # Pad with zeros
            pad_size = MAX_EDGE_NUM - original_H
            net_4d = torch.cat([net_4d, torch.zeros(1, net_4d.shape[1], pad_size, 1, device=net_4d.device, dtype=net_4d.dtype)], dim=2)
            inp_4d = torch.cat([inp_4d, torch.zeros(1, inp_4d.shape[1], pad_size, 1, device=inp_4d.device, dtype=inp_4d.dtype)], dim=2)
            corr_4d = torch.cat([corr_4d, torch.zeros(1, corr_4d.shape[1], pad_size, 1, device=corr_4d.device, dtype=corr_4d.dtype)], dim=2)
            ii_4d = torch.cat([ii_4d, torch.zeros(1, 1, pad_size, 1, device=ii_4d.device, dtype=ii_4d.dtype)], dim=2)
            jj_4d = torch.cat([jj_4d, torch.zeros(1, 1, pad_size, 1, device=jj_4d.device, dtype=jj_4d.dtype)], dim=2)
            kk_4d = torch.cat([kk_4d, torch.zeros(1, 1, pad_size, 1, device=kk_4d.device, dtype=kk_4d.dtype)], dim=2)
        elif original_H > MAX_EDGE_NUM:
            # Slice to MAX_EDGE_NUM
            net_4d = net_4d[:, :, :MAX_EDGE_NUM, :]
            inp_4d = inp_4d[:, :, :MAX_EDGE_NUM, :]
            corr_4d = corr_4d[:, :, :MAX_EDGE_NUM, :]
            ii_4d = ii_4d[:, :, :MAX_EDGE_NUM, :]
            jj_4d = jj_4d[:, :, :MAX_EDGE_NUM, :]
            kk_4d = kk_4d[:, :, :MAX_EDGE_NUM, :]
        
        # Convert to numpy and ensure correct dtypes
        # Note: ONNX models expect inputs on CPU, but we'll move outputs back to GPU
        inputs_dict = {}
        if net_4d.is_cuda:
            inputs_dict[input_names[0]] = net_4d.cpu().numpy().astype(np.float32)
            inputs_dict[input_names[1]] = inp_4d.cpu().numpy().astype(np.float32)
            inputs_dict[input_names[2]] = corr_4d.cpu().numpy().astype(np.float32)
            inputs_dict[input_names[3]] = ii_4d.cpu().numpy().astype(np.int32)
            inputs_dict[input_names[4]] = jj_4d.cpu().numpy().astype(np.int32)
            inputs_dict[input_names[5]] = kk_4d.cpu().numpy().astype(np.int32)
        else:
            inputs_dict[input_names[0]] = net_4d.numpy().astype(np.float32)
            inputs_dict[input_names[1]] = inp_4d.numpy().astype(np.float32)
            inputs_dict[input_names[2]] = corr_4d.numpy().astype(np.float32)
            inputs_dict[input_names[3]] = ii_4d.numpy().astype(np.int32)
            inputs_dict[input_names[4]] = jj_4d.numpy().astype(np.int32)
            inputs_dict[input_names[5]] = kk_4d.numpy().astype(np.int32)
        
        # Run inference
        outputs = self.update_session.run(output_names, inputs_dict)
        
        # Convert outputs back to torch tensors and move to GPU
        net_out = torch.from_numpy(outputs[0])  # [1, DIM, MAX_EDGE_NUM, 1]
        d_out = torch.from_numpy(outputs[1])    # [1, 2, MAX_EDGE_NUM, 1]
        w_out = torch.from_numpy(outputs[2])    # [1, 2, MAX_EDGE_NUM, 1]
        
        if torch.cuda.is_available():
            net_out = net_out.cuda()
            d_out = d_out.cuda()
            w_out = w_out.cuda()
        
        # Slice back to original H if needed
        if original_H < MAX_EDGE_NUM:
            net_out = net_out[:, :, :original_H, :]
            d_out = d_out[:, :, :original_H, :]
            w_out = w_out[:, :, :original_H, :]
        elif original_H > MAX_EDGE_NUM:
            # If we sliced, we need to handle this case (shouldn't happen often)
            # For now, just use what we got
            pass
        
        # Convert back to [1, H, DIM] format for compatibility
        net_out = net_out.squeeze(-1).permute(0, 2, 1)  # [1, DIM, H, 1] -> [1, H, DIM]
        d_out = d_out.squeeze(-1).permute(0, 2, 1)      # [1, 2, H, 1] -> [1, H, 2]
        w_out = w_out.squeeze(-1).permute(0, 2, 1)      # [1, 2, H, 1] -> [1, H, 2]
        
        return net_out, (d_out, w_out, None)

