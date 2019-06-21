import numpy as np
from mlreco.utils.gnn.cluster import form_clusters_new
from mlreco.utils.gnn.compton import filter_compton
from mlreco.visualization.voxels import scatter_label
from mlreco.utils.gnn.primary import assign_primaries3

def find_shower_cone(dbscan, groups, em_primaries, energy_data, length_factor=4.3, slope_percentile=55, slope_factor=6.1, verbose=False):
    """
    dbscan: data parsed from "dbscan_label": ["parse_dbscan", "sparse3d_fivetypes"]
    groups: data parsed from "group_label": ["parse_cluster3d_clean", "cluster3d_mcst", "sparse3d_fivetypes"]
    em_primaries: data parsed from "em_primaries" : ["parse_em_primaries", "sparse3d_data", "particle_mcst"]
    energy_data: data parsed from "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    
    returns a list of length len(em_primaries) containing np arrays, each of which contains the indices corresponding to the voxels in the cone of the corresponding EM primary
    """
    clusts = form_clusters_new(dbscan)
    assigned_primaries = assign_primaries3(em_primaries, clusts, groups)
    selected_voxels = []
    
    for i in range(len(assigned_primaries)):
        if assigned_primaries[i] != -1:
            if verbose:
                print('------')
            p = em_primaries[i]
            em_point = p[:3]

            c = clusts[assigned_primaries[i]]
            
            # find primary cluster axis
            primary_points = dbscan[c][:, :3]
            primary_energies = energy_data[c][:, -1]
            primary_center = np.average(primary_points.T, axis=1, weights=primary_energies)
            primary_axis = primary_center - em_point

            # find furthest particle from cone axis
            primary_length = np.linalg.norm(primary_axis)
            direction = primary_axis / primary_length
            axis_distances = np.linalg.norm(np.cross(primary_points-primary_center, primary_points-em_point), axis=1)/primary_length
            axis_projections = np.dot(primary_points - em_point, direction)
            primary_slope = np.percentile(axis_distances/axis_projections, slope_percentile)
            if verbose:
                print('primary cluster half-length', primary_length)
                print('primary cluster selected cone angle', np.arctan(primary_slope)/np.pi*180)

            
            # define a cone around the primary axis
            cone_length = length_factor * primary_length
            cone_slope = slope_factor * primary_slope
            cone_vertex = em_point
            cone_axis = direction

            classified_indices = []
            for j in range(len(dbscan)):
                point = dbscan[j]
                coord = point[:3]
                axis_dist = np.dot(coord - em_point, cone_axis)
                if 0 <= axis_dist and axis_dist <= cone_length:
                    cone_radius = axis_dist * cone_slope
                    point_radius = np.linalg.norm(np.cross(coord-(em_point + cone_axis), coord-em_point))
                    if point_radius < cone_radius:
                        # point inside cone
                        classified_indices.append(j)
            classified_indices = np.array(classified_indices)
            selected_voxels.append(classified_indices)
        else:
            selected_voxels.append([])

    return selected_voxels