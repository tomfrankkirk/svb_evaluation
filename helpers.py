import numpy as np 
import pyvista as pv 
import sys 

sys.path.insert(1, '/Users/thomaskirk/modules/svb_module')
from svb.main import run
from svb.data import HybridModel

sys.path.insert(1, '/Users/thomaskirk/modules/svb_models_asl')
from svb_models_asl import AslRestModel 

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

def pv_plot(surface, data, **kwargs):
    pl = pv.Plotter(window_size=(600, 400))
    faces = 3 * np.ones((surface.tris.shape[0], 4), dtype=int)
    faces[:,1:] = surface.tris 
    mesh = pv.PolyData(surface.points, faces=faces).rotate_z(240, inplace=True)
    pl.add_mesh(mesh, scalars=data, **kwargs)
    pl.show(jupyter_backend='panel')

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def make_activation(sph): 

    az, el, r = cart2sph(*(sph.points - sph.points.mean(0)).T)

    scale = 6
    ctx_cbf = 40 + 20 * (np.sin(scale * el)**2 + np.sin(scale/2 * az)**2)

    mask = ctx_cbf * np.ones_like(ctx_cbf)
    mask[np.abs(el) > (0.40) * np.pi] = 60
    mask[np.abs(el) > 0.49 * np.pi] = 80
    high = mask > 79
    low = mask < 41

    rounds = 6
    adj = sph.adjacency_matrix()
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

    rounds = 5
    cbf_smooth = cbf * np.ones_like(cbf)
    for _ in range(rounds): 
        cbf_smooth = ((adj @ cbf_smooth) + cbf_smooth) / (1 + adj.sum(1).A.flatten())

    return cbf_smooth


def simulate_data(noise, rpt, ctx_cbf, proj, plds):

    data = np.empty((*proj.spc.size, len(plds) * rpt), dtype=np.int8)
    data_model = HybridModel(proj.spc.make_nifti(data), projector=proj)
    asl_model = AslRestModel(data_model,
            plds=plds, repeats=rpt, casl=True)

    with tf.Session() as sess:
        cbf = np.concatenate([
                ctx_cbf, 
                20 * np.ones(data_model.n_vol_nodes),
                60 * np.ones(data_model.n_roi_nodes)  ]) 

        att = np.concatenate([
                1.3 * np.ones(data_model.n_surf_nodes), 
                1.6 * np.ones(data_model.n_vol_nodes),
                1.3 * np.ones(data_model.n_roi_nodes)  ]) 

        data = sess.run(asl_model.evaluate(
                [ cbf[:,None].astype(np.float32), 
                  att[:,None].astype(np.float32) ], 
                  asl_model.tpts()))

    data = (data_model.n2v_coo @ data).reshape(*proj.spc.size, -1)
    data += np.random.normal(0, noise, size=data.shape)
    return data