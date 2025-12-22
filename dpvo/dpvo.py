import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *
from .ba import python_ba_wrapper

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:
    """
    Dense Patch Visual Odometry (DPVO) system class.

    Args:
        cfg (Namespace): Configuration object containing system parameters, e.g.,
            - PATCHES_PER_FRAME (int)
            - BUFFER_SIZE (int)
            - MIXED_PRECISION (bool)
            - LOOP_CLOSURE (bool)
            - CLASSIC_LOOP_CLOSURE (bool)
            - MAX_EDGE_AGE (int)
        network (str or Path): Path to pre-trained network weights.
        ht (int, optional): Input image height. Default is 480.
        wd (int, optional): Input image width. Default is 640.
        viz (bool, optional): Enable visualization viewer. Default is False.

    Attributes:
        is_initialized (bool): Flag indicating if DPVO has been initialized.
        enable_timing (bool): Flag to enable timing measurement.
        M (int): Number of patches per frame.
        N (int): Buffer size (maximum number of frames).
        ht, wd (int): Image height and width.
        DIM (int): Feature dimension for patch descriptors (from cfg).
        RES (int): Resolution factor for downsampling.
        tlist (list): List storing timestamps or other temporal info.
        counter (int): Frame counter.
        ran_global_ba (np.ndarray): Boolean array tracking global bundle adjustment calls.
        image_ (torch.Tensor): Dummy image for visualization (ht x wd x 3).
        kwargs (dict): Device and dtype arguments for tensors.
        pmem (int): Patch memory size.
        mem (int): Frame memory size.
        last_global_ba (int): Index of last global BA call (if loop closure enabled).
        imap_ (torch.Tensor): Local feature memory for patches (pmem x M x DIM).
        gmap_ (torch.Tensor): Global feature memory for patches (pmem x M x 128 x P x P).
        pg (PatchGraph): PatchGraph object storing frames, patches, and edges.
        fmap1_, fmap2_ (torch.Tensor): Feature pyramid maps at different resolutions.
        pyramid (tuple): Tuple containing fmap1_ and fmap2_.
        viewer: Visualization viewer object (if viz enabled).
    """
    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()
        
        self._pg_shape_ref = None
        self.show_config()
            
    def show_config(self):
        """ Pretty config + runtime state printout with emojis """

        print("\n====================== 🟣 DPVO CONFIG 🟣 ======================\n")

        # -------------------- Network Info -------------------- #
        print("📦 NETWORK")
        if hasattr(self, "network"):
            net_name = self.network.__class__.__name__
        else:
            net_name = "Not Loaded"

        print(f"   🧠 Model:              {net_name}")
        print(f"   🎛 DIM:                {self.DIM}")
        print(f"   🔍 Patch Size (RES):   {self.RES}x{self.RES}")
        print(f"   🧩 Descriptors (P):    {self.P}")

        if hasattr(self, "_loaded_checkpoint") and self._loaded_checkpoint is not None:
            print(f"   📁 Checkpoint:         {self._loaded_checkpoint}")
        else:
            print(f"   📁 Checkpoint:         <unknown or external model>")

        print("\n---------------------- Runtime ---------------------------\n")

        print(f"📸 Input Resolution:     {self.ht} × {self.wd}")
        print(f"🧵 Torch Threads:        {torch.get_num_threads()}")
        print(f"⚙️ Mixed Precision:      {self.cfg.MIXED_PRECISION}")
        print(f"🚦 Initialized:          {self.is_initialized}")
        print(f"⏱ Timing Enabled:       {self.enable_timing}")

        print("\n---------------------- Memory ----------------------------\n")

        print(f"🧠 Frame Buffer Size (N): {self.N}")
        print(f"🧩 Patches Per Frame (M): {self.M}")
        print(f"📁 Patch Memory (pmem):   {self.pmem}")
        print(f"📁 Frame Memory (mem):    {self.mem}")

        print(f"📈 Global BA Calls:       {np.sum(self.ran_global_ba)}")

        print("\n---------------------- Feature Maps ----------------------\n")

        print(f"🗺 iMap Shape:            {tuple(self.imap_.shape)}")
        print(f"🎛 Corr Map Shape:        {tuple(self.gmap_.shape)}")

        print("\n🔎 Feature Pyramids:")
        print(f"   🟦 fmap1:              {tuple(self.fmap1_.shape)}")
        print(f"   🟥 fmap2:              {tuple(self.fmap2_.shape)}")

        print("\n---------------------- Loop Closure ----------------------\n")

        print(f"🔄 Loop Closure Enabled:  {self.cfg.LOOP_CLOSURE}")
        print(f"🏛 Classic Backend:       {self.cfg.CLASSIC_LOOP_CLOSURE}")

        print("\n===========================================================\n")

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        if indicies is not None:
            ii, jj = indicies
        else:
            # Slice to active edges only
            num_active = self.pg.num_edges
            ii = self.pg.kk[:num_active]
            jj = self.pg.jj[:num_active]
        
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        if indicies is not None:
            ii, jj, kk = indicies
        else:
            # Slice to active edges only
            num_active = self.pg.num_edges
            ii = self.pg.ii[:num_active]
            jj = self.pg.jj[:num_active]
            kk = self.pg.kk[:num_active]
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        # (ii,jj) : (patch_indices, frame_indices) pairs
        """
        FOR EXAMPLE :
            Before append_factors():
            num_edges = 5
            Arrays: [ii: 0,1,2,3,4, 0,0,0,0,0, ...]  (max_edges=10)
                    [jj: 1,2,3,4,5, 0,0,0,0,0, ...]
                    [kk: 10,20,30,40,50, 0,0,0,0,0, ...]
                    [net: [state1], [state2], ..., [state5], [0], [0], ...]
                    
            Call: append_factors(ii=[60, 70], jj=[6, 6])
                - ii=[60, 70] = patch indices 60 and 70
                - jj=[6, 6] = both patches observed in frame 6
                - self.ix[60] and self.ix[70] = source frame indices (e.g., frame 4)
            
            After append_factors():
            num_edges = 7
            Arrays: [ii: 0,1,2,3,4, 4,4, 0,0,0, ...]  ← Added source frames
                    [jj: 1,2,3,4,5, 6,6, 0,0,0, ...]  ← Added target frames
                    [kk: 10,20,30,40,50, 60,70, 0,0,0, ...]  ← Added patch indices
                    [net: [s1], [s2], ..., [s5], [0], [0], [0], [0], ...]  ← Initialized new states
        """
        num_new = len(ii)
        if self.pg.num_edges + num_new > self.pg.max_edges:
            raise RuntimeError(
                f"Maximum edges ({self.pg.max_edges}) exceeded. "
                f"Current: {self.pg.num_edges}, Adding: {num_new}. "
                f"Increase MAX_EDGES in config."
            )
        
        start_idx = self.pg.num_edges
        end_idx = start_idx + num_new
        
        self.pg.jj[start_idx:end_idx] = jj
        self.pg.kk[start_idx:end_idx] = ii
        self.pg.ii[start_idx:end_idx] = self.ix[ii]

        # Initialize new net entries to zero
        self.pg.net[:, start_idx:end_idx] = torch.zeros(1, num_new, self.DIM, **self.kwargs)
        
        self.pg.num_edges = end_idx

    def remove_factors(self, m, store: bool):
        # m is a boolean mask - can be [num_edges] or [max_edges], we'll handle both
        num_active = self.pg.num_edges
        
        # If mask is larger than active edges, slice it
        if m.shape[0] > num_active:
            m = m[:num_active]
        elif m.shape[0] < num_active:
            # Pad mask if smaller (shouldn't happen, but be safe)
            m_padded = torch.zeros(num_active, dtype=torch.bool, device=m.device)
            m_padded[:m.shape[0]] = m
            m = m_padded
        
        if store:
            num_to_store = m.sum().item()
            if num_to_store > 0:
                if self.pg.num_edges_inac + num_to_store > self.pg.max_edges:
                    # If inactive buffer is full, just skip storing (or could truncate oldest)
                    print(f"Warning: Inactive edge buffer full, not storing {num_to_store} edges")
                else:
                    start_inac = self.pg.num_edges_inac
                    end_inac = start_inac + num_to_store
                    
                    # Only store active edges that are being removed
                    self.pg.ii_inac[start_inac:end_inac] = self.pg.ii[:num_active][m]
                    self.pg.jj_inac[start_inac:end_inac] = self.pg.jj[:num_active][m]
                    self.pg.kk_inac[start_inac:end_inac] = self.pg.kk[:num_active][m]
                    self.pg.weight_inac[:, start_inac:end_inac] = self.pg.weight[:, :num_active][:, m]
                    self.pg.target_inac[:, start_inac:end_inac] = self.pg.target[:, :num_active][:, m]
                    
                    self.pg.num_edges_inac = end_inac
        
        # Compact active edges by moving remaining edges to fill gaps
        keep_mask = ~m
        num_keep = keep_mask.sum().item()
        
        if num_keep > 0 and num_keep < num_active:
            # Move kept edges to the front
            self.pg.ii[:num_keep] = self.pg.ii[:num_active][keep_mask]
            self.pg.jj[:num_keep] = self.pg.jj[:num_active][keep_mask]
            self.pg.kk[:num_keep] = self.pg.kk[:num_active][keep_mask]
            self.pg.net[:, :num_keep] = self.pg.net[:, :num_active][:, keep_mask]
            self.pg.weight[:, :num_keep] = self.pg.weight[:, :num_active][:, keep_mask]
            self.pg.target[:, :num_keep] = self.pg.target[:, :num_active][:, keep_mask]
        
        self.pg.num_edges = num_keep

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        # Only check active edges
        num_active = self.pg.num_edges
        active_ii = self.pg.ii[:num_active]
        active_jj = self.pg.jj[:num_active]
        k = (active_ii == i) & (active_jj == j)
        if k.sum() == 0:
            return 0.0
        ii = active_ii[k]
        jj = active_jj[k]
        kk = self.pg.kk[:num_active][k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        """
        Before keyframe removal (n=10, KEYFRAME_INDEX=4):
        Frames: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                                ↑  ↑  ↑
                                i  k  j  (checking frame 6)

        If motion between frames 5↔7 is small:
        - Frame 6 is redundant
        - Remove frame 6
        - Shift frames 7,8,9 → positions 6,7,8

        After removal (n=9):
        Frames: [0, 1, 2, 3, 4, 5, 6, 7, 8]
                                   ↑  ↑  ↑
                                (was 7) (was 8) (was 9)
        """

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        # bidirectional motion magnitude (average of both directions)
        m = self.motionmag(i, j) + self.motionmag(j, i) # optical flow magnitude from frame i to j & add j to i
 
        if m / 2 < self.cfg.KEYFRAME_THRESH: 
            # the candidate frame is redundant and can be removed
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            # Only check active edges
            num_active = self.pg.num_edges
            active_ii = self.pg.ii[:num_active]
            active_jj = self.pg.jj[:num_active]
            to_remove = (active_ii == k) | (active_jj == k)
            # Pad to full size for remove_factors
            to_remove_full = torch.zeros(self.pg.max_edges, dtype=torch.bool, device="cuda")
            to_remove_full[:num_active] = to_remove
            self.remove_factors(to_remove_full, store=False) # store=False means these edges are not moved to inactive storage

            # Update indices for remaining active edges
            num_active = self.pg.num_edges
            if num_active > 0:
                # Create views for easier indexing
                active_ii = self.pg.ii[:num_active]
                active_jj = self.pg.jj[:num_active]
                active_kk = self.pg.kk[:num_active]
                
                mask_ii = active_ii > k
                mask_jj = active_jj > k
                # Update in-place on the actual tensor slices
                active_kk[mask_ii] -= self.M
                active_ii[mask_ii] -= 1
                active_jj[mask_jj] -= 1

            for i in range(k, self.n-1):
                # Shifts all frame data backward to fill the gap left by frame k
                # Updates timestamps, colors, poses, patches, intrinsics, and feature maps
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]
            # Step 8: Update Counters (Lines 481-482)
            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        # Only check active edges
        num_active = self.pg.num_edges
        active_kk = self.pg.kk[:num_active]
        active_ii = self.pg.ii[:num_active]
        active_jj = self.pg.jj[:num_active]
        
        to_remove = self.ix[active_kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((active_jj - active_ii) > 30) & (active_jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        
        # Pad to full size for remove_factors
        to_remove_full = torch.zeros(self.pg.max_edges, dtype=torch.bool, device="cuda")
        to_remove_full[:num_active] = to_remove
        self.remove_factors(to_remove_full, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        num_active = self.pg.num_edges
        num_inac = self.pg.num_edges_inac
        total_edges = num_active + num_inac
        
        # Concatenate active and inactive edges (sliced to actual sizes)
        full_target = torch.cat((self.pg.target_inac[:, :num_inac], self.pg.target[:, :num_active]), dim=1)
        full_weight = torch.cat((self.pg.weight_inac[:, :num_inac], self.pg.weight[:, :num_active]), dim=1)
        full_ii = torch.cat((self.pg.ii_inac[:num_inac], self.pg.ii[:num_active]))
        full_jj = torch.cat((self.pg.jj_inac[:num_inac], self.pg.jj[:num_active]))
        full_kk = torch.cat((self.pg.kk_inac[:num_inac], self.pg.kk[:num_active]))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = full_ii.min().item() if total_edges > 0 else 0
        if total_edges > 0:
            fastba.BA(self.poses, self.patches, self.intrinsics,
                full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    
    def _check_pg_static_shape(self):
        # With static shapes, these should always be the same
        shapes = (
            tuple(self.pg.ii.shape),
            tuple(self.pg.jj.shape),
            tuple(self.pg.kk.shape),
        )

        if self._pg_shape_ref is None:
            self._pg_shape_ref = shapes
            print(f"[PG SHAPE INIT] ii/jj/kk shapes = {shapes} (static, num_edges={self.pg.num_edges})")
        else:
            # Shapes should be static now, but check anyway
            # if shapes != self._pg_shape_ref:
            print(
                "[PG SHAPE CHANGE DETECTED - UNEXPECTED!]\n"
                f"  previous = {self._pg_shape_ref}\n"
                f"  current  = {shapes}"
            )
            self._pg_shape_ref = shapes
            # Log active edge count for debugging
            if self.pg.num_edges != getattr(self, '_last_num_edges', 0):
                print(f"[PG ACTIVE EDGES] {self.pg.num_edges}/{self.pg.max_edges}")
                self._last_num_edges = self.pg.num_edges
    
    
    def update(self):
        num_active = self.pg.num_edges
        if num_active == 0:
            return
            
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                self._check_pg_static_shape()
                corr = self.corr(coords)
                # Slice to active edges only
                active_kk = self.pg.kk[:num_active]
                active_ii = self.pg.ii[:num_active]
                active_jj = self.pg.jj[:num_active]
                ctx = self.imap[:, active_kk % (self.M * self.pmem)]
                
                # Slice net to active edges for update
                active_net = self.pg.net[:, :num_active]
                active_net, (delta, weight, _) = \
                    self.network.update(active_net, ctx, corr, None, active_ii, active_jj, active_kk)
                # Write back to full tensor
                self.pg.net[:, :num_active] = active_net

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        # Store target and weight (sliced to active edges)
        self.pg.target[:, :num_active] = target
        self.pg.weight[:, :num_active] = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (active_ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    # fastba.BA(self.poses, self.patches, self.intrinsics, 
                    #     target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
                    '''Use python torch BA function from ba.py, Alister 2025-12-15 updated'''
                    new_poses = python_ba_wrapper(self.poses, self.patches, self.intrinsics, target, weight,
                                        lmbda, active_ii, active_jj, active_kk, PPF=None, t0=t0, t1=self.n, iterations=1, eff_impl=False)
                    
                    # print(type(new_poses))
                    # print(dir(new_poses))
                    # print(self.__dict__.keys())
        
                    self.poses.copy_(new_poses)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        """
        What it does:
            Selects old patches: patches from frames (n-r) to (n-1)
                t0 = M * (n - r): first patch index in the range
                t1 = M * (n - 1): last patch index in the range
                Connects to new frame: frame n-1 (the frame just added)
            Returns: (patch_indices, frame_indices) pairs
            Example:
                Assume:
                    n = 10 (current frame count, frame 9 is the newest)
                    M = 48 (patches per frame)
                    PATCH_LIFETIME = 6
                Calculation:
                    t0 = 48 * (10 - 6) = 48 * 4 = 192 (first patch from frame 4)
                    t1 = 48 * (10 - 1) = 48 * 9 = 432 (last patch from frame 9)
                    Patches: [192, 193, ..., 431] (240 patches = 5 frames × 48)
                    Target frame: [9] (frame 9)
                Result:
                    Creates 240 edges connecting patches [192-431] → frame 9
                    Each old patch is observed in the new frame
        
        Visual:
            Frames:  [4]  [5]  [6]  [7]  [8]  [9] ← new frame
            Patches: 192  240  288  336  384  432
                      ↓    ↓    ↓    ↓    ↓
                      └────┴────┴────┴────┴────→ Frame 9
                    (All old patches connected to new frame)
        """
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        """
        What it does:
            1. Selects new patches: patches from frame n-1 (the newest frame)
                - t0 = M * (n - 1): first patch of frame n-1
                - t1 = M * n: first patch of frame n (exclusive)
            2. Connects to old frames: frames from max(n-r, 0) to n-1
            3. Returns: (patch_indices, frame_indices) pairs
            Example:
                Same setup:
                n = 10, M = 48, PATCH_LIFETIME = 6
                Calculation:
                    - t0 = 48 * (10 - 1) = 432 (first patch of frame 9)
                    - t1 = 48 * 10 = 480 (first patch of frame 10, exclusive)
                New patches: [432, 433, ..., 479] (48 patches from frame 9)
                Target frames: [4, 5, 6, 7, 8, 9] (last 6 frames)
                Result:
                    Creates 48 × 6 = 288 edges connecting new patches [432-479] → frames [4-9]
                    Each new patch is observed in multiple old frames
        Visual:
            New Patches (Frame 9):  432, 433, ..., 479 (48 patches)
                                     ↓    ↓   ...  ↓
            Old Frames:             [4] [5] [6] [7] [8] [9]
                                (Each new patch → all old frames)
        """
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        image = 2 * (image[None,None] / 255.0) - 0.5
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        """
        Why Both Are Needed
           1. Forward edges: track old patches into the new frame
           2. Backward edges: track new patches back into old frames
            Together they create bidirectional constraints for bundle adjustment.
        Example :
            After calling both functions with n=10, M=48, r=6:

                Forward edges:  240 edges (old patches → frame 9)
                Backward edges: 288 edges (new patches → old frames)
                Total:          528 new edges added
        """
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()
        """
        Summary
        If keyframe() is not called:
        Frames accumulate → crash at BUFFER_SIZE
        Edges accumulate → crash at MAX_EDGES or severe slowdown
        Performance degrades → slower BA and correlation
        Memory grows → potential OOM
        keyframe() is essential for maintaining bounded memory and performance. 
        It should be called after every update() when the system is initialized (as it currently is on line 734).
        """

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
