from typing import Iterator, List, NamedTuple

import h5py
import torch
from torch import Tensor

class AttnData(NamedTuple):
    attn: Tensor
    self_attn: Tensor
    instr_id: str
    txt_instr: str
    instr: Tensor

    scan: str
    viewpoint: str
    agent_heading: float
    views: List[str]
    objs: List[str]
    obj_orients: Tensor
    view_orients: Tensor

def ds_iterator(file_path: str) -> Iterator[h5py.Group]:
    with h5py.File(file_path) as reader:
        instr_id = list(reader.keys())[0]
        while True:
            return_instr_id = yield reader[instr_id]
            if return_instr_id is not None:
                instr_id = return_instr_id

def step_iterator(ds: h5py.Group) -> Iterator[AttnData]:
    for step_id in ds.keys():
        step_ds = ds[step_id]

        attn = torch.from_numpy(step_ds['attn'][:])
        self_attn = torch.from_numpy(step_ds['self_attn'][:])

        yield AttnData(
            attn=attn,
            self_attn=self_attn,
            instr_id=ds.attrs['instr_id'],
            txt_instr=ds.attrs['txt_instr'],
            instr=ds['instr'][:],
            scan=step_ds.attrs['scan'],
            viewpoint=step_ds.attrs['viewpoint'],
            agent_heading=step_ds.attrs['agent_heading'],
            views=step_ds.attrs['views'],
            objs=step_ds.attrs['objs'],
            obj_orients=torch.from_numpy(step_ds['obj_orients'][:]),
            view_orients=torch.from_numpy(step_ds['view_orients'][:])
        )


        
