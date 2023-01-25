import sys
import os.path as op

from svb.data import DataModel
from svb_models_asl import AslRestModel
from svb_models_asl.aslrest import TIME_SCALE

import numpy as np
import toblerone as tob 
import pyvista as pv 
    
np.random.seed(1)

PLDS = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]) * TIME_SCALE

GM_ATT = 1.1 * TIME_SCALE
WM_ATT = 1.8 * TIME_SCALE
GM_CBF = 60 
WM_CBF = 20 

ROI_NAMES = [ 'T1_first-L_Accu_first', 'T1_first-L_Amyg_first', 'T1_first-L_Caud_first', 'T1_first-L_Hipp_first', 'T1_first-L_Pall_first', 'T1_first-L_Puta_first', 'T1_first-L_Thal_first' ]
SUBCORT_CBF = np.array([45, 50, 55, 60, 65, 70, 75])

def make_activation(sph): 

    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    az, el, _ = cart2sph(*(sph.points - sph.points.mean(0)).T)


    scale = 6
    ctx_cbf = 40 + 20 * (np.sin(scale * el)**2 + np.sin(scale/2 * az)**2)
    adj = sph.adjacency_matrix()

    mask = ctx_cbf * np.ones_like(ctx_cbf)
    mask[np.abs(el) > (0.40) * np.pi] = 60
    mask[np.abs(el) > 0.49 * np.pi] = 80
    high = mask > 79
    low = mask < 41

    rounds = 6
    for _ in range(rounds): 
        for idx in np.flatnonzero(high): 
            neighbours = adj[idx,:].indices
            high[neighbours] = 1 

        for idx in np.flatnonzero(low): 
            neighbours = adj[idx,:].indices
            low[neighbours] = 1 

    cbf = 60 * np.ones_like(ctx_cbf)
    cbf[high] = 80 
    cbf[low] = 40 

    mask = ctx_cbf * np.ones_like(ctx_cbf)
    mask[np.abs(el) > (0.40) * np.pi] = 60
    mask[np.abs(el) > 0.49 * np.pi] = 80
    high = mask > 79
    low = mask < 41

    rounds = 20
    cbf_smooth = cbf * np.ones_like(cbf)
    for _ in range(rounds): 
        cbf_smooth = ((adj @ cbf_smooth) + cbf_smooth) / (1 + adj.sum(1).A.flatten())

    return cbf_smooth



def simulate_data(proj, noise, rpt, ctx_cbf=None):
    data = np.zeros((proj.spc.size.prod(), len(PLDS) * rpt), dtype=np.float32)
    data_model = DataModel(proj.spc.make_nifti(data), mode='hybrid', projector=proj)
    asl_model = AslRestModel(data_model,
            plds=PLDS, repeats=rpt, casl=True)

    if ctx_cbf is None: 
        ctx_cbf = 60 * np.ones(data_model.n_surf_nodes)
    else: 
        ctx_cbf = ctx_cbf[data_model.surf_nodes]

    cbf = np.concatenate([
            ctx_cbf, 
            WM_CBF * np.ones(data_model.n_vol_nodes),
            SUBCORT_CBF 
                ]) 

    att = np.concatenate([
            GM_ATT * np.ones(data_model.n_surf_nodes), 
            WM_ATT * np.ones(data_model.n_vol_nodes),
            GM_ATT * np.ones(data_model.n_roi_nodes) 
                ]) 

    data = asl_model.test_voxel_data(
        params={ 'ftiss': cbf.astype(np.float32), 
                 'delttiss': att.astype(np.float32) }, 
        tpts=asl_model.tpts(), 
        noise_sd=noise, 
        masked=False)

    return data, data_model


def pv_plot(surface, data, **kwargs):
    pl = pv.Plotter(window_size=(600, 400))
    faces = 3 * np.ones((surface.tris.shape[0], 4), dtype=int)
    faces[:,1:] = surface.tris 
    mesh = pv.PolyData(surface.points, faces=faces).rotate_z(240, inplace=True)
    pl.add_mesh(mesh, scalars=data, **kwargs)
    pl.show(jupyter_backend='pythreejs')