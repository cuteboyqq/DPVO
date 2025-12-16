import torch
from dpvo.net import UpdateONNX, neighbors  # ONNX-compatible class + neighbors function

# ---------------------------
# 1️⃣ Load pretrained weights
# ---------------------------
print("📥 Loading dpvo.pth ...")
dpvo_state = torch.load("dpvo.pth", map_location="cpu")
print("✅ dpvo.pth loaded successfully!")

# Extract only Update parameters, ignoring lmbda.* keys
update_state_dict = {
    k.replace("module.update.", ""): v
    for k, v in dpvo_state.items()
    if k.startswith("module.update.") and "lmbda" not in k
}
print(f"📝 Extracted {len(update_state_dict)} Update parameters")

# Initialize UpdateONNX with matching p and dim
update_onnx = UpdateONNX(p=3)
update_onnx.load_state_dict(update_state_dict, strict=False)
update_onnx.eval()
print("🤖 UpdateONNX model initialized and ready")

# ---------------------------
# 2️⃣ Create dummy inputs
# ---------------------------
DIM = 384
B, N, p = 1, 1024, 3
print(f"🎲 Creating dummy inputs with B={B}, N={N}, DIM={DIM}, p={p} ...")
net = torch.randn(B, N, DIM)
inp = torch.randn(B, N, DIM)
corr = torch.randn(B, N, 2*49*p*p)
ii = torch.randint(0, 100, (N,))
jj = torch.randint(0, 100, (N,))
kk = torch.randint(0, 100, (N,))
# Precompute neighbors using kk and jj
ix, jx = neighbors(kk, jj)

# ---------------------------
# 3️⃣ Export to ONNX
# ---------------------------
print("🚀 Exporting UpdateONNX model to ONNX format ...")
update_onnx.forward = lambda *args: UpdateONNX.forward(update_onnx, *args, export_onnx=True)

torch.onnx.export(
    update_onnx,
    (net, inp, corr, None, ii, jj, kk),
    "update.onnx",
    input_names=["net","inp","corr","flow","ii","jj","kk"],
    output_names=["net_out","d_out","w_out"],
    opset_version=12,
    dynamic_axes={
        "net": {0: "B", 1: "N"},
        "inp": {0: "B", 1: "N"},
        "corr": {0: "B", 1: "N"},
        "ii": {0: "N"},
        "jj": {0: "N"},
        "kk": {0: "N"},
        "net_out": {0: "B", 1: "N"},
        "d_out": {0: "B", 1: "N"},
        "w_out": {0: "B", 1: "N"},
    }
)
print("✅ update.onnx exported successfully! 🎉")
