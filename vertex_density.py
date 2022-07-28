# %%
import sys 

sys.path.insert(1, '/Users/thomaskirk/modules/svb_module')
from svb.main import run
from svb.data import HybridModel

sys.path.insert(1, '/Users/thomaskirk/modules/svb_models_asl')
from svb_models_asl import AslRestModel 

import numpy as np
import toblerone as tob 
import subprocess

import matplotlib.pyplot as plt 
from matplotlib.cm import get_cmap
cmap = np.array(get_cmap('tab10').colors)
import seaborn as sns 
import helpers 
import nibabel as nib 

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

# %%
sph32 = tob.Surface('sph_32k.surf.gii')
cbf_truth = helpers.make_activation(sph32)
# helpers.pv_plot(sph, cbf_truth, clim=[40, 80]) 

# %%
proj32 = tob.Projector.load('/Users/thomaskirk/Modules/svb_module/brain_proj_3.h5')
PLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# %%
# Generate projectors at each vertex resolution (5k -> 32k) at 3mm iso 
# Measure number of vertices per voxel 
# Run SVB to estimate CBF on each resolution 
# Calculate MSE wrt 32k ground truth

lps_32_path = 'vertex_density/lps_32k.surf.gii'
lws_32_path = 'vertex_density/lws_32k.surf.gii'
sph_32_path = 'vertex_density/sph_32k.surf.gii'

proj32['LPS'].save(lps_32_path)
proj32['LWS'].save(lws_32_path)

densities = np.arange(4,36,4)

# %%
# for x in densities: 

#     s = f"vertex_density/sph_{x}k.surf.gii"
#     cmd = f"wb_command -surface-create-sphere {x*1000:d} {s}"
#     subprocess.run(cmd, shell=True)

#     lps = f'vertex_density/lps_{x}k.surf.gii'
#     cmd = f"surface_resample {lps_32_path} vertex_density/sph_32k.surf.gii {x*1000:d} {lps}"
#     subprocess.run(cmd, shell=True)

#     lws = f'vertex_density/lws_{x}k.surf.gii'
#     cmd = f"surface_resample {lws_32_path} vertex_density/sph_32k.surf.gii {x*1000:d} {lws}"
#     subprocess.run(cmd, shell=True)

#     hemi = tob.Hemisphere(lws, lps, 'L')
    # p = tob.Projector(hemi, proj_base.spc)
    # p.save(f'vertex_density/proj_{x}k.h5')

# %%
data_mean = nib.load('/Users/thomaskirk/Modules/svb_module/hybrid_test/input_data.nii.gz')

options = {
    "mode": "hybrid", 

    "learning_rate" : 0.1,
    # "lr_decay_rate": 1.0, 
    "sample_size": 10,
    "ss_increase_factor": 2, 
    "epochs": 1500,

    "batch_size" : len(PLDS),
    "log_stream" : sys.stdout,
    "plds": PLDS, 
    "repeats": 1, 
    "casl": True, 
    "prior_type": "M",
    "display_step": 100, 
    "debug": True, 

}

for x in densities: 

    # p = tob.Projector.load(f'vertex_density/proj_{x}k.h5')    
    p = proj32 
    mask = (p.pvs()[...,:2] > 0.05).any(-1)

    runtime, svb, training_history = run(
        data_mean, "aslrest", f"vertex_density/svb_{x}k", 
        projector=p, mask=mask, **options)

# %%
for x in densities: 
    inpath = f"vertex_density/svb_{x}k/mean_ftiss_L_cortex.func.gii"
    outpath = f"vertex_density/svb_{x}k/mean_ftiss_L_cortex_32k.func.gii"
    insph = f"vertex_density/sph_{x}k.surf.gii"
    outsph = f"vertex_density/sph_32k.surf.gii"
    cmd = f"wb_command -metric-resample {inpath} {insph} {outsph} BARYCENTRIC {outpath}"
    subprocess.run(cmd, shell=True)

# %%
cbf_svb = []
thickness = np.linalg.norm(proj_base['LPS'].points - proj_base['LWS'].points, ord=2)
thick_mask = thickness > 0.5

for x in densities: 
    inpath = f"vertex_density/svb_{x}k/mean_ftiss_L_cortex_32k.func.gii"
    c = nib.load(inpath).darrays[0].data
    cbf_svb.append(c)

# %%
helpers.pv_plot(sph, cbf_svb[2], clim=[40, 80]) 


# %%
ssd = [ ((c - cbf_truth)[thick_mask] ** 2).sum for c in cbf_svb ]
plt.plot(densities, ssd)


