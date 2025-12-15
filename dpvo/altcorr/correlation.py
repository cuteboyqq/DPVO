import torch
import cuda_corr
from .correlation_kernel import ( 
patchify_python_forward,
patchify_python_backward, 
corr_cuda_forward, 
corr_backward_kernel, 
corr_cuda_backward,
)


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, ii, jj, radius, dropout):
        """ forward correlation """
        ctx.save_for_backward(fmap1, fmap2, coords, ii, jj)
        ctx.radius = radius
        ctx.dropout = dropout
        
        # Print shapes and parameter values for debugging
        # print("=== CorrLayer.forward Debug ===")
        # print(f"fmap1 shape: {tuple(fmap1.shape)}")
        # print(f"fmap2 shape: {tuple(fmap2.shape)}")
        # print(f"coords shape: {tuple(coords.shape)}")
        # print(f"ii shape: {tuple(ii.shape)}, min={ii.min().item()}, max={ii.max().item()}")
        # print(f"jj shape: {tuple(jj.shape)}, min={jj.min().item()}, max={jj.max().item()}")
        # print(f"radius: {radius}")
        # print(f"dropout: {dropout}")
        # print("===============================")
        corr, = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
        # corr, = corr_cuda_forward(fmap1, fmap2, coords, ii, jj, radius)

        return corr

    @staticmethod
    def backward(ctx, grad):
        """ backward correlation """
        fmap1, fmap2, coords, ii, jj = ctx.saved_tensors

        if ctx.dropout < 1:
            perm = torch.rand(len(ii), device="cuda") < ctx.dropout
            coords = coords[:,perm]
            grad = grad[:,perm]
            ii = ii[perm]
            jj = jj[perm]

        # fmap1_grad, fmap2_grad = \
        #     cuda_corr.backward(fmap1, fmap2, coords, ii, jj, grad, ctx.radius)
        fmap1_grad, fmap2_grad = \
            corr_cuda_backward(fmap1, fmap2, coords, ii, jj, grad, ctx.radius)

        return fmap1_grad, fmap2_grad, None, None, None, None, None


class PatchLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, net, coords, radius):
        """ forward patchify """
        ctx.radius = radius
        ctx.save_for_backward(net, coords)
        
        # patches, = cuda_corr.patchify_forward(net, coords, radius)
        patches, = patchify_python_forward(net, coords, radius)
        return patches

    @staticmethod
    def backward(ctx, grad):
        """ backward patchify """
        net, coords = ctx.saved_tensors
        # grad, = cuda_corr.patchify_backward(net, coords, grad, ctx.radius)
        grad, = patchify_python_backward(net, coords, grad, ctx.radius)

        return grad, None, None

def patchify(net, coords, radius, mode='bilinear'):
    """ extract patches """

    patches = PatchLayer.apply(net, coords, radius)

    if mode == 'bilinear':
        offset = (coords - coords.floor()).to(net.device)
        dx, dy = offset[:,:,None,None,None].unbind(dim=-1)

        d = 2 * radius + 1
        x00 = (1-dy) * (1-dx) * patches[...,:d,:d]
        x01 = (1-dy) * (  dx) * patches[...,:d,1:]
        x10 = (  dy) * (1-dx) * patches[...,1:,:d]
        x11 = (  dy) * (  dx) * patches[...,1:,1:]

        return x00 + x01 + x10 + x11

    return patches
    

def corr(fmap1, fmap2, coords, ii, jj, radius=1, dropout=1):
    return CorrLayer.apply(fmap1, fmap2, coords, ii, jj, radius, dropout)



#-------------------Alister add 2025-12-11-----------------------------------------------------------
# Python version of the C++ extension wrapper

def patchify_forward(net: torch.Tensor, coords: torch.Tensor, radius: int):
    """
    Forward patchify wrapper.

    Args:
        net    : Tensor, [B, C, H, W] or similar
        coords : Tensor, [B, M, 2, H, W] or similar
        radius : int, search radius

    Returns:
        patches : Tensor or tuple of Tensors
    """
    # Call the underlying CUDA function (assume already implemented in Python/CUDA)
    patches = patchify_cuda_forward(net, coords, radius)
    # Return as tuple for consistency with C++ std::vector
    return (patches,)


def patchify_backward(net: torch.Tensor, coords: torch.Tensor, grad: torch.Tensor, radius: int):
    """
    Backward patchify wrapper.

    Args:
        net    : Tensor
        coords : Tensor
        grad   : Tensor, gradient of output
        radius : int

    Returns:
        grad_net : Tensor or tuple
    """
    grad_net = patchify_cuda_backward(net, coords, grad, radius)
    return (grad_net,)
