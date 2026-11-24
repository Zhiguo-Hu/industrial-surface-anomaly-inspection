"""Two frozen inspection members; use the refined member for raw localization."""
from torch import nn


class StripEnsemble(nn.Module):
    def __init__(self,members):
        super().__init__()
        if len(members)!=2:raise ValueError('Exactly two inspection members required')
        self.members=nn.ModuleList(members)
