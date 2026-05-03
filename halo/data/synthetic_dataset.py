# halo/data/synthetic_dataset.py
import torch
from torch.utils.data import Dataset
from halo.config import HaloConfig


class SyntheticMultimodalDataset(Dataset):
    """Paired (text_embed, image_embed) synthetic dataset.

    Construction:
        1. text_embed ~ mixture of 5 Gaussians in R^text_embed_dim
        2. image_embed = A @ text_embed + noise
           where A is a fixed random orthogonal-like projection
           and noise ~ N(0, 0.1 * I)

    This creates a correlated cross-modal pair.
    """

    def __init__(
        self,
        n_samples: int,
        cfg: HaloConfig,
        seed: int = 42,
    ) -> None:
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)

        d_text  = cfg.text_embed_dim
        d_image = cfg.image_embed_dim

        # 5-component Gaussian mixture means
        n_components = 5
        means = torch.randn(n_components, d_text, generator=rng)

        # Fixed random projection matrix A (text -> image), shape (d_image, d_text)
        A_raw = torch.randn(d_image, d_text, generator=rng)
        # Orthonormalise columns for stable projection
        A, _ = torch.linalg.qr(A_raw.T)
        self.A = A.T  # (d_image, d_text)

        # Sample component assignments and text embeddings
        comp = torch.randint(0, n_components, (n_samples,), generator=rng)
        text_embeds = means[comp] + 0.5 * torch.randn(n_samples, d_text, generator=rng)

        # Project to image space + noise
        image_embeds = (self.A @ text_embeds.T).T  # (n_samples, d_image)
        image_embeds = image_embeds + 0.1 * torch.randn(n_samples, d_image, generator=rng)

        self.text_embeds  = text_embeds.float()
        self.image_embeds = image_embeds.float()

    def __len__(self) -> int:
        return len(self.text_embeds)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text_embed":  self.text_embeds[idx],
            "image_embed": self.image_embeds[idx],
        }
