import numpy as np
import torch as pt
import numpy.typing as npt
from typing import Generator
from dataclasses import dataclass, fields,field


@dataclass
class Structure:
    xyz: npt.NDArray[np.float32]
    names: npt.NDArray[np.str_]
    elements: npt.NDArray[np.str_]
    resnames: npt.NDArray[np.str_]
    resids: npt.NDArray[np.int32]
    chain_names: npt.NDArray[np.str_]
    resids_ndb: npt.NDArray[np.int32] = None
    charges: npt.NDArray[np.float32] = None
    active_site: npt.NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    
    def __post_init__(self):
        # Initialize charges with zeros if not provided
        if self.charges is None or len(self.charges) == 0:
            # Use the length of xyz (assuming xyz is always populated)
            n_atoms = len(self.xyz)
            self.charges = np.zeros(n_atoms, dtype=np.float32)
    
    def __iter__(self):
        for field in fields(self):
            yield field.name

    def __getitem__(self, idx):
        return Structure(**{key: getattr(self, key)[idx] for key in self})
    
@dataclass
class StructureData:
    X: pt.Tensor
    qe: pt.Tensor
    qr: pt.Tensor
    qn: pt.Tensor
    qc: pt.Tensor
    ac: pt.Tensor
    Mr: pt.Tensor
    Mc: pt.Tensor

    def to(self, device):
        self.X.to(device)
        self.qe.to(device)
        self.qr.to(device)
        self.qn.to(device)
        self.qc.to(device)
        self.ac.to(device)
        self.Mr.to(device)
        self.Mc.to(device)

    def cpu(self):
        self.to(pt.device("cpu"))

    def __getitem__(self, idx):
        return StructureData(
            X=self.X[idx],
            qe=self.qe[idx],
            qr=self.qr[idx],
            qn=self.qn[idx],
            qc=self.qc[idx],
            ac=self.ac[idx],
            Mr=self.Mr[idx][:, pt.sum(self.Mr[idx], dim=0) > 0.5],
            Mc=self.Mc[idx][:, pt.sum(self.Mc[idx], dim=0) > 0.5],
        )

    def __iter__(self) -> Generator[pt.Tensor, None, None]:
        for field in fields(self):
            yield getattr(self, field.name)
