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

# Initialize UpdateONNX with matching p
p = 3
update_onnx = UpdateONNX(p=p,export_onnx=True)
update_onnx.load_state_dict(update_state_dict, strict=False)
update_onnx.eval()
print("🤖 UpdateONNX model initialized and ready")


# ---------------------------
# 2️⃣ Create dummy inputs (4D for CV28)
# ---------------------------
B, H, DIM = 1, 1024, 384

net  = torch.randn(B, DIM, H, 1)
inp  = torch.randn(B, DIM, H, 1)
corr = torch.randn(B, 2*49*p*p, H, 1)

ii = torch.randint(0, 100, (H,1), dtype=torch.int32)
jj = torch.randint(0, 100, (H,1), dtype=torch.int32)
kk = torch.randint(0, 100, (H,1), dtype=torch.int32)

ii = ii.view(1, -1, 1)
jj = jj.view(1, -1, 1)
kk = kk.view(1, -1, 1)


# Precompute neighbors if needed
# ix, jx = neighbors(kk, jj)

# ---------------------------
# 3️⃣ Reshape for CV28 (4D) inputs
# ---------------------------


# ---------------------------
# 4️⃣ Export to ONNX
# ---------------------------
print("🚀 Exporting UpdateONNX model to ONNX format ...")
RUN = False
if RUN:
    torch.onnx.export(
        update_onnx,
        (net, inp, corr, ii, jj, kk),
        "update.onnx",
        input_names=["net", "inp", "corr", "ii", "jj", "kk"],
        output_names=["net_out", "d_out", "w_out"],
        opset_version=12,
        dynamic_axes={
            "net":  {2: "H"},
            "inp":  {2: "H"},
            "corr": {2: "H"},

            "ii": {0: "H"},
            "jj": {0: "H"},
            "kk": {0: "H"},

            "net_out": {2: "H"},
            "d_out":   {2: "H"},
            "w_out":   {2: "H"},
        }
        # dynamic_axes={
        #     "net":  {0: "B", 2: "H"},
        #     "inp":  {0: "B", 2: "H"},
        #     "corr": {0: "B", 2: "H"},

        #     "ii": {0: "H"},
        #     "jj": {0: "H"},
        #     "kk": {0: "H"},

        #     "net_out": {0: "B", 2: "H"},
        #     "d_out":   {0: "B", 2: "H"},
        #     "w_out":   {0: "B", 2: "H"},
        # }

    )
RUN2 = True
if RUN2:
    torch.onnx.export(
        update_onnx,
        (net, inp, corr, ii, jj, kk),   # already 1D indices
        "update_int32_static.onnx",
        input_names=["net","inp","corr","ii","jj","kk"],
        output_names=["net_out","d_out","w_out"],
        dynamic_axes=None,  # ❌ Do NOT use dynamic axes for CV28
        # do_constant_folding=True,
        opset_version=12
    )

print("✅ update.onnx exported successfully! 🎉")


# ---------------------------
# 4️⃣ Verify shapes
# ---------------------------
import onnx
model = onnx.load("update_int32_static.onnx")
for inp in model.graph.input:
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    print(f"Input '{inp.name}' shape: {shape}")
for out in model.graph.output:
    shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    print(f"Output '{out.name}' shape: {shape}")