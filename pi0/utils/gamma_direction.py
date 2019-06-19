'''
This module contains methods associated with inferring the direction of a gamma shower

'''

def do_calculation(data, start, length=5.):
    '''
    Calculates the best fit direction of the gamma shower based on a PCA of the
    shower start
    Args:
        data - a numpy array Nx5 containing hits associated with gamma shower
        start - x,y,z of shower start (``tuple``)
        length - the length of shower to include in the PCA (cm)
    Returns:
        a numpy array Nx7 containing [event, [x,y,z,batch,x_comp,y_comp,z_comp]]. XX_comp are the components of the unit vectors determined by the PCA
    
    '''
    # select hits within ``length`` of shower start
    # perform PCA on those hits
    pass