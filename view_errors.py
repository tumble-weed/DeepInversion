class ViewErrors():
    def __init__(self,NVIEW,len_al_buffer,N_AL_TRAIN):
        len_al_buffer = NVIEW*iterations_per_layer*(1 if skipfirst else 2)
        if N_AL_TRAIN is not None:
            len_al_buffer = N_AL_TRAIN
        self.view_errors = {'error':np.zeros((len_al_buffer,)),
                        'azim':np.zeros((len_al_buffer,)),
                        'elev':np.zeros((len_al_buffer,)),
                        'dist':np.zeros((len_al_buffer,)),
                        'n':0
                        }        
        
    
    pass