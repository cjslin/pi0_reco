import numpy as np
from mlreco.utils.gnn.cluster import form_clusters_new
from mlreco.utils.gnn.compton import filter_compton
from mlreco.visualization.voxels import scatter_label
from mlreco.utils.gnn.primary import assign_primaries3

# TODO make sure entire primary cluster is assigned even if it's outside the cone

def find_shower_cone(dbscan, groups, em_primaries, energy_data, types, length_factor=14.107334041, slope_percentile=52.94032412, slope_factor=5.86322059, return_truth=False, verbose=False):
    """
    dbscan: data parsed from "dbscan_label": ["parse_dbscan", "sparse3d_fivetypes"]
    groups: data parsed from "group_label": ["parse_cluster3d_clean", "cluster3d_mcst", "sparse3d_fivetypes"]
    em_primaries: data parsed from "em_primaries" : ["parse_em_primaries", "sparse3d_data", "particle_mcst"]
    energy_data: data parsed from "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    
    returns a list of length len(em_primaries) containing np arrays, each of which contains the indices corresponding to the voxels in the cone of the corresponding EM primary
    """
    clusts = form_clusters_new(dbscan)
    assigned_primaries = assign_primaries3(em_primaries, clusts, groups).astype(int)
    selected_voxels = []
    true_voxels = []
    for i in range(len(assigned_primaries)):
        if assigned_primaries[i] != -1:
            if verbose:
                print('------')
            c = clusts[assigned_primaries[i]]
            
            if return_truth:
                group_ids = np.unique(groups[c][:, -1])
                type_id = -1
                for g in groups[c]:
                    for j in range(len(types)):
                        if np.array_equal(g[:3], types[j][:3]):
                            type_id = types[j][-1]
                            break
                    if type_id != -1:
                        break
                true_indices = np.where(np.logical_and(np.isin(groups[:, -1], group_ids), types[:, -1] >= 2))[0]
                true_voxels.append(true_indices)

            p = em_primaries[i]
            em_point = p[:3]

            # find primary cluster axis
            primary_points = dbscan[c][:, :3]
            primary_energies = energy_data[c][:, -1]
            if np.sum(primary_energies) == 0:
                selected_voxels.append(np.array([]))
                continue
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
                point = types[j]
                if point[-1] < 2:
                    continue
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
            if return_truth:
                true_voxels.append(np.array([]))
            selected_voxels.append(np.array([]))
    
    if return_truth:
        return true_voxels, selected_voxels
    else:
        return selected_voxels