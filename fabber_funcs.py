
import subprocess 
import numpy as np 
import os.path as op 
import regtricks as rt 
import os 
import nibabel
import sys 
import copy 
from svb.main import run
import shutil

def basil_cmd(asl, mask, opts, odir, pvs=None):
    
    if op.exists(odir): 
        shutil.rmtree(odir)
    os.makedirs(odir, exist_ok=True)
    spc = rt.ImageSpace(mask)

    if isinstance(pvs, str): 
        gpath = op.join(odir, 'pvgm.nii.gz')
        wpath = op.join(odir, 'pvwm.nii.gz')
        pvs = nibabel.load(pvs).get_fdata()
        for path,arr in zip([gpath,wpath], [pvs[...,0],pvs[...,1]]):
            spc.save_image(arr,path)

    elif isinstance(pvs, list):
        gpath, wpath = pvs 

    optsfile = op.join(odir, 'basil_opts.txt')
    with open(optsfile, 'w') as f: 
        f.write("--save-noise-mean\n")
        if not 'bat' in opts: 
            f.write("--fixbat\n--fixbatwm\n--casl\n")
        for key,val in opts.items():
            if val is not True: 
                f.write(f"--{key}={val}\n")
            else: 
                f.write(f"--{key}\n")

        f.write("\n")

    odir = op.join(odir, 'basil_out')
    cmd = f"basil -i {asl} -o {odir} -m {mask}"
    cmd += f" --optfile={optsfile} "
    if pvs is not None: 
        cmd += f" --pgm={gpath} --pwm={wpath}"

    return cmd 

def basil_surface(asl, mask, opts, odir, pvs, projector):

    cmd = basil_cmd(asl, mask, opts, odir, pvs)
    print(cmd)
    subprocess.run(cmd, shell=True)

    gmftiss = op.join(odir, 'basil_out/step2/mean_ftiss.nii.gz')
    gmftiss = nibabel.load(gmftiss).get_fdata() 
    gmftiss_surf = projector.vol2surf(gmftiss.flatten(), edge_scale=False)
    return gmftiss_surf

def oxasl_cmd(asl, mask, odir, opts, pvs, projector):
    
    os.makedirs(odir, exist_ok=True)

    mpath = op.join(odir, 'mask.nii.gz')
    apath = op.join(odir, 'asl.nii.gz')
    gpath = op.join(odir, 'pvgm.nii.gz')
    wpath = op.join(odir, 'pvwm.nii.gz')
    
    projector.spc.save_image(asl, apath)
    projector.spc.save_image(mask, mpath)    
    projector.spc.save_image(pvs[...,0], gpath)
    projector.spc.save_image(pvs[...,1], wpath)

    optspath = op.join(odir, 'bopts.txt')
    opts = copy.deepcopy(opts)
    cmd = [
        "oxasl", "-i", apath, "-m", mpath, "-o", odir, "--iaf", "diff",  
        "--fit-options", optspath, "--casl", "--artoff", '--ibf', opts.pop('ibf'), 
        "--debug", "--no-report", "--overwrite", "--bolus", str(opts.pop('bolus')), 
    ] 

    plds = opts.pop('plds')
    cmd.append('--plds')
    cmd.append(",".join(str(p) for p in plds))

    with open(optspath, 'w') as f: 
        for key,val in opts.items():
            if val is not True: 
                f.write(f"--{key}={val}\n")
            else: 
                f.write(f"--{key}\n")

        f.write("\n")

    cmd += ["--pvcorr", "--pvgm", gpath, "--pvwm", wpath,]

    print(cmd)
    subprocess.run(cmd)

    gmftiss = op.join(odir, 'output_pvcorr/native/perfusion.nii.gz')
    gmftiss = nibabel.load(gmftiss).get_fdata() 
    gmftiss_surf = projector.vol2surf(gmftiss.flatten(), edge_scale=False)
    return gmftiss_surf


if __name__ == '__main__': 
    
    pass
    data = 'hybrid_test/simdata.nii.gz'
    mask = data
    opts = { 'bolus': 1.8, 'repeats': 5, 'ti1': 1.25 }
    odir = 'hybrid_test/basil_surface'
    cmd = basil_cmd(data, mask, opts, odir)
    print(cmd)