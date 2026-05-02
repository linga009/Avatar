import os
import numpy as np
import torch
from torch.utils.data import Dataset
from med_rai.utils.se3_utils import se3_log

TRAJ_HORIZON = 10
IMG_SIZE = 224
FT_NOISE_STD = 0.01

_SPLITS = {
    "train": ["B", "C", "D", "E", "F", "G", "H"],
    "val":   ["A"],
}

class JIGSAWSDataset(Dataset):
    """
    Loads JIGSAWS kinematic + gesture data.
    Force/torque is simulated as Gaussian noise (JIGSAWS has no F/T sensors).
    Returns one sample = one frame with a TRAJ_HORIZON-step kinematic look-ahead.
    """

    def __init__(self, root: str, task: str = "Knot_Tying",
                 split: str = "train", seq_len: int = 8):
        self.root = root
        self.task = task
        self.seq_len = seq_len
        subjects = _SPLITS[split]
        self.samples = self._load_samples(subjects)

    def _load_samples(self, subjects):
        samples = []
        kin_dir = os.path.join(self.root, self.task, "kinematics", "AllGestures")
        trn_dir = os.path.join(self.root, self.task, "transcriptions")
        if not os.path.exists(kin_dir):
            return samples
        for fname in sorted(os.listdir(kin_dir)):
            if not fname.endswith(".txt"):
                continue
            # Extract subject letter: e.g. "Knot_Tying_B001.txt" -> "B"
            parts = fname.replace(".txt", "").split("_")
            subject = parts[-1][0]
            if subject not in subjects:
                continue
            kin_path = os.path.join(kin_dir, fname)
            trn_path = os.path.join(trn_dir, fname)
            kin = np.loadtxt(kin_path)
            if kin.ndim == 1:
                kin = kin[np.newaxis, :]
            gestures = self._load_gestures(trn_path, len(kin))
            for i in range(len(kin) - TRAJ_HORIZON):
                samples.append((kin, gestures, i))
        return samples

    def _load_gestures(self, path, n_frames):
        labels = np.zeros(n_frames, dtype=int)
        if not os.path.exists(path):
            return labels
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start = int(parts[0])
                    end = min(int(parts[1]), n_frames)
                    g = int(parts[2][1:]) - 1  # G1->0, G15->14
                    labels[start:end] = g
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kin, gestures, i = self.samples[idx]
        # PSM1 end-effector: columns 0:3=position, 3:12=rotation matrix (row-major)
        t_vec = torch.tensor(kin[i, 0:3], dtype=torch.float32)
        R_mat = torch.tensor(kin[i, 3:12], dtype=torch.float32).reshape(3, 3)

        # Trajectory: next TRAJ_HORIZON frames in se(3)
        traj_xi = []
        for j in range(i, i + TRAJ_HORIZON):
            p_j = torch.tensor(kin[j, 0:3], dtype=torch.float32)
            R_j = torch.tensor(kin[j, 3:12], dtype=torch.float32).reshape(3, 3)
            xi_j = se3_log(R_j.unsqueeze(0), p_j.unsqueeze(0)).squeeze(0)
            traj_xi.append(xi_j)
        xi_traj = torch.stack(traj_xi, dim=0)  # (H, 6)

        ft = torch.randn(6) * FT_NOISE_STD
        image = torch.randn(3, IMG_SIZE, IMG_SIZE)
        token_ids = torch.zeros(self.seq_len, dtype=torch.long)

        return {
            "images":         image,
            "R":              R_mat,
            "t":              t_vec,
            "ft":             ft,
            "token_ids":      token_ids,
            "xi_traj":        xi_traj,
            "gesture_labels": torch.tensor(int(gestures[i]), dtype=torch.long),
            "text_targets":   torch.zeros(self.seq_len + 3, dtype=torch.long),
        }
