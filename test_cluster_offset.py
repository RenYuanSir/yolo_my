import types
import sys
import torch
from loss_fix import compute_ranking_loss_fixed

# Provide dummy flash_attn module so dataset can be imported without full deps
flash_attn = types.ModuleType('flash_attn')
flash_attn.flash_attn_interface = types.ModuleType('flash_attn_interface')
flash_attn.flash_attn_interface.flash_attn_func = lambda *a, **k: None
sys.modules.setdefault('flash_attn', flash_attn)
sys.modules.setdefault('flash_attn.flash_attn_interface', flash_attn.flash_attn_interface)

from ultralytics.data.dataset import TomatoYOLODataset

class Dummy:
    device = 'cpu'
    margin = 0.1

def test_cluster_offset_collation():
    """Clusters from different images remain separate after collation."""
    s1 = {'cluster_ids': torch.tensor([[1]]),
          'h_rel': torch.tensor([[0.1]]),
          'cls': torch.tensor([[2]])}
    s2 = {'cluster_ids': torch.tensor([[1]]),
          'h_rel': torch.tensor([[0.8]]),
          'cls': torch.tensor([[0]])}

    collated = TomatoYOLODataset.collate_fn([s1, s2])

    # After collation IDs from each image should be offset
    ids = collated['cluster_ids'].view(-1).tolist()
    assert ids[0] != ids[1], 'cluster ids should be unique across images'

    loss_offset = compute_ranking_loss_fixed(Dummy(), collated)

    # Manually concatenate without offset to simulate wrong behavior
    manual = {
        'cluster_ids': torch.cat([s1['cluster_ids'], s2['cluster_ids']], 0),
        'h_rel': torch.cat([s1['h_rel'], s2['h_rel']], 0),
        'cls': torch.cat([s1['cls'], s2['cls']], 0),
    }
    loss_no_offset = compute_ranking_loss_fixed(Dummy(), manual)

    # Loss with proper offset should stay near minimal value
    assert loss_offset < loss_no_offset
