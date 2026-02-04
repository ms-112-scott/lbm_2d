import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
from lbm2d_mrt_les import LBM2D_MRT_LES


if __name__ == "__main__":
    config_path = "src/lbm_mrt_les/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = utils.load_config(config_path)
    mask = utils.create_mask(config)
    solver = LBM2D_MRT_LES(config, mask_data=mask)
    solver.solve()
