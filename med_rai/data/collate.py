from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

def medrai_collate(samples: List[Dict]) -> Dict:
    """Collates MedRAI samples, padding variable-length token sequences."""
    batch = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if key in ("token_ids", "text_targets"):
            batch[key] = pad_sequence(vals, batch_first=True, padding_value=0)
        else:
            batch[key] = torch.stack(vals)
    return batch
