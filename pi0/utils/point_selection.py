'''
This module contains methods associated with selecting start points of gamma showers

'''

def do_selection(data, label=2):
    '''
    Args:
        data - a numpy array Nx5 containing proposed points
        label - label number for shower-like points
        
    Returns:
        a pruned numpy array nx5 containing selected points
        
    '''
    mask = data[:, -1] == label
    return data[mask]