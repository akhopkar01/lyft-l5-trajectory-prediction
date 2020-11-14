from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
import numpy as np

zarr_dt = ChunkedDataset("/home/majoradi/Documents/l5-sample/sample.zarr")
zarr_dt.open()

# additional information is required for rasterisation
cfg = load_config_data("~/")
rast = build_rasterizer(cfg, LocalDataManager("/tmp/l5kit_data"))

# create a mask where an agent every 100th is set to True
agents_mask = np.zeros(len(zarr_dt.agents), dtype=np.bool)
agents_mask[np.arange(0, len(agents_mask), 100)] = True


dataset = AgentDataset(cfg, zarr_dt, rast, agents_mask=agents_mask)