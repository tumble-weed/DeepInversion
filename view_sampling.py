from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
)                      
from pytorch3d.renderer import cameras as p3dcameras
import os
import colorful
import numpy as np
def sample_view_params(
                      pr_model,
                      iteration_loc,
                       iteration,
                       trends,
                       device,
                       ENABLE_MESH,
                       N_AL_TRAIN,
                       view_errors,
                       UNIFORM_AZIM=None,
                       ):

    if os.environ.get('USE_FIXED_DIR',False) == '1' and True:
        # eye_at = (0,0,2)
        # R, T = look_at_view_transform(eye = [eye_at],at=((0,0,0,),))
        # elev_range = (-45,45)
        # azim_range = (-90,90)
        # # dist_range = (2,5)
        # dist_range = (1,)                        
        elev = 0
        azim = 0
        dist = 1
        R, T = look_at_view_transform(dist=dist,azim=azim,elev=elev,at=((0,0,0,),))
    else:
        if False:
            dist = 2
            dirn = (np.random.random(3) - 0.5)*2
            unit_dirn = dirn/np.linalg.norm(dirn)
            eye_at = unit_dirn*dist
            R, T = look_at_view_transform(eye = [eye_at],at=((0,0,0,),))
            R,T = [R],[T]
            NVIEW = 1                            
        else:
            elev_range = (-45,45)
            if "ELEV_MAG" in os.environ:
                elev_mag = int(os.environ['ELEV_MAG'])
                elev_range = (-elev_mag,elev_mag)                            
            azim_range = (-90,90)
            if "AZIM_MAG" in os.environ:
                azim_mag = int(os.environ['AZIM_MAG'])
                azim_range = (-azim_mag,azim_mag)                            
            """
            if "DIST_MAG" in os.environ:
                dist_mag = float(os.environ['DIST_MAG'])
                dist_range = (1,dist_mag)                                                 
                
            """  
            dist_min,dist_max = 1,2
            if "DIST_MIN" in os.environ:
                dist_min = float(os.environ['DIST_MIN'])
            if "DIST_MAX" in os.environ:
                dist_max = float(os.environ['DIST_MAX'])
            if ENABLE_MESH == 1:
                print(colorful.red("hard coding dist_range"))
                (dist_min,dist_max) = (1.2,1.5)
            if os.environ.get('ADA_CAM_DIST',False) == '1':
                assert pr_model.vertsparam.shape[-1] == 3
                max_point_distance_from_origin = (pr_model.vertsparam).norm(dim=1).max()
                dist_min = dist_min * max_point_distance_from_origin.item()
                dist_max = dist_max * max_point_distance_from_origin.item()
                if running_dist_min is not None:
                    dist_min = running_dist_min *0.9 + 0.1*dist_min
                    dist_max = running_dist_max *0.9 + 0.1*dist_max
                running_dist_min = dist_min
                running_dist_max = dist_max
                trends['dist_min'].append(dist_min)
                trends['dist_max'].append(dist_max)
            dist_range = (dist_min,dist_max)
            NVIEW = int(os.environ.get('NVIEW',1))         
            if  os.environ.get('ANNEAL',False) == '1':
                if iteration < 2000:
                    azim_range = (-5,5)
                elif iteration < 4000 and azim_mag >= 15:
                    azim_range = (-15,15)
                elif iteration < 6000  and azim_mag >= 30:
                    azim_range = (-30,30)
                elif iteration < 8000  and azim_mag >= 45:
                    azim_range = (-45,45)
                elif iteration < 10000  and azim_mag >= 60:
                    azim_range = (-60,60)
                elif iteration < 12000 and azim_mag >= 75:
                    azim_range = (-75,75)
                elif iteration < 14000 and azim_mag >= 90:
                    azim_range = (-90,90)
                
            dist = dist_range[0] + (dist_range[1]-dist_range[0])*np.random.random(NVIEW)
            dist = dist_range[0] *np.ones(dist.shape)
            print(colorful.red("setting dist to the min"))
            # dist = 2
            # print(colorful.red(f"setting dist to {dist}"))
            elev = elev_range[0] + (elev_range[1]-elev_range[0])*np.random.random(NVIEW)
            # UNIFORM_AZIM = os.environ.get('UNIFORM_AZIM',False) == '1'
            # UNIFORM_AZIM = 'UNIFORM_AZIM' in os.environ
            assert UNIFORM_AZIM in [True,False]
            if not UNIFORM_AZIM:
                azim = azim_range[0] + (azim_range[1]-azim_range[0])*np.random.random(NVIEW)
            elif UNIFORM_AZIM:
                # import colorful
                print(colorful.red("azim periodic"))
                azim0 = (azim_range[1]-azim_range[0])*np.random.random()
                if os.environ.get('AZIM0','') != '':
                    azim0 = int(os.environ['AZIM0'])
                azim = azim0 + (360/NVIEW)*np.arange(NVIEW).astype(np.float32)
                # bring into 0-360 range
                azim = azim % 360
                # import ipdb; ipdb.set_trace()
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            if os.environ.get('USE_VIEW_SAMPLING',False) == '1':
                if (iteration_loc > 0): 
                    active_learner = None
                    
                    if (iteration_loc % 2) == 1:
                        X_for_oversampler = np.stack([
                            view_errors['elev'][:view_errors['n']],
                            view_errors['azim'][:view_errors['n']],
                            view_errors['dist'][:view_errors['n']]],axis=1)
                        y_for_oversampler = view_errors['error'][:view_errors['n']]
                        X_for_oversampler = X_for_oversampler[-N_AL_TRAIN:]
                        y_for_oversampler = y_for_oversampler[-N_AL_TRAIN:]
                        # import ipdb; ipdb.set_trace()
                        from view_sampling_al import GaussianProcessActiveLearner
                        active_learner = GaussianProcessActiveLearner(X_for_oversampler,y_for_oversampler,device=device)
                        active_learner.fit(10)
                    if active_learner is not None:
                        n_oversample = 1000
                        elev_oversample = elev_range[0] + (elev_range[1]-elev_range[0])*np.random.random(n_oversample)
                        azim_oversample = azim_range[0] + (azim_range[1]-azim_range[0])*np.random.random(n_oversample)                                
                        dist_oversample = dist_range[0] + (dist_range[1]-dist_range[0])*np.random.random(n_oversample)

                        X_star = np.stack([
                            elev_oversample,
                            azim_oversample,
                            dist_oversample
                        ],axis=1)
                        sampled = active_learner.sample(X_star,NVIEW,device=device)
                        elev = sampled[:,0]
                        azim = sampled[:,1]
                        dist = sampled[:,2]                                
    return elev, azim, dist