import numpy as np
import os

# ---------------------------
# 1️⃣ Configuration
# ---------------------------
net_out_dir = "./dummy/dummy_net"
os.makedirs(net_out_dir, exist_ok=True)

inp_out_dir = "./dummy/dummy_inp"
os.makedirs(inp_out_dir, exist_ok=True)

corr_out_dir = "./dummy/dummy_corr"
os.makedirs(corr_out_dir, exist_ok=True)

ii_out_dir = "./dummy/dummy_ii"
os.makedirs(ii_out_dir, exist_ok=True)

jj_out_dir = "./dummy/dummy_jj"
os.makedirs(jj_out_dir, exist_ok=True)

kk_out_dir = "./dummy/dummy_kk"
os.makedirs(kk_out_dir, exist_ok=True)

B, N, DIM = 1, 1024, 384
p = 3

# ---------------------------
# 2️⃣ Generate dummy inputs
# ---------------------------

# net feature map
net = np.random.randn(B, N, DIM).astype(np.float32)
net.tofile(os.path.join(net_out_dir, "net.bin"))

# inp feature map
inp = np.random.randn(B, N, DIM).astype(np.float32)
inp.tofile(os.path.join(inp_out_dir, "inp.bin"))

# corr
corr = np.random.randn(B, N, 2*49*p*p).astype(np.float32)
corr.tofile(os.path.join(corr_out_dir, "corr.bin"))

# indices (int64)
ii = np.random.randint(0, 100, size=(N,), dtype=np.int32)
jj = np.random.randint(0, 100, size=(N,), dtype=np.int32)
kk = np.random.randint(0, 100, size=(N,), dtype=np.int32)

ii.tofile(os.path.join(ii_out_dir, "ii.bin"))
jj.tofile(os.path.join(jj_out_dir, "jj.bin"))
kk.tofile(os.path.join(kk_out_dir, "kk.bin"))

# ---------------------------
# 3️⃣ Done
# ---------------------------
print(f"✅ Dummy .bin files saved in {net_out_dir}")
print("Shapes:")
print(f"net: {net.shape}, inp: {inp.shape}, corr: {corr.shape}")
print(f"ii: {ii.shape}, jj: {jj.shape}, kk: {kk.shape}")
