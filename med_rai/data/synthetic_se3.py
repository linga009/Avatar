import torch
from torch.utils.data import Dataset
from med_rai.utils.se3_utils import se3_log, se3_exp

TRAJ_HORIZON = 10

class SyntheticSE3Dataset(Dataset):
    """
    Generates synthetic geodesic trajectories on SE(3) for RFM head warmup.
    Each sample is a straight-line path in se(3) between two random poses.
    This directly implements the Flow Matching training target: linear interpolation.
    """

    def __init__(self, n_samples: int = 10000, t_scale: float = 0.05):
        self.n_samples = n_samples
        self.t_scale = t_scale
        # Pre-generate all samples for reproducibility
        torch.manual_seed(42)
        self.data = [self._generate() for _ in range(n_samples)]

    def _generate(self):
        # Random SE(3) poses via QR decomposition (no pytorch3d needed)
        R0 = torch.linalg.qr(torch.randn(3, 3))[0]
        t0 = torch.randn(3) * self.t_scale
        R1 = torch.linalg.qr(torch.randn(3, 3))[0]
        t1 = torch.randn(3) * self.t_scale

        xi_0 = se3_log(R0.unsqueeze(0), t0.unsqueeze(0)).squeeze(0)
        xi_1 = se3_log(R1.unsqueeze(0), t1.unsqueeze(0)).squeeze(0)

        traj = []
        for step in range(TRAJ_HORIZON):
            t = (step + 1) / TRAJ_HORIZON
            traj.append((1 - t) * xi_0 + t * xi_1)

        return {"xi_0": xi_0, "xi_1": xi_1, "xi_traj": torch.stack(traj, dim=0)}

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
