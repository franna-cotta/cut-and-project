import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, delaunay_plot_2d, distance_matrix, KDTree
from collections import defaultdict, OrderedDict
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from itertools import combinations
from cut_and_project_utility import pts_in_ndim_interval, shoelace_area, recover_edges, tri_similar
from FranGraph import powerDiagram

# Not working right now
def plot_scipy_KDTree(plot, outer_pts, proj_region : list = [], fig_size : tuple = (18, 18)):
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        pts = pts_in_ndim_interal(outer_pts, proj_region)
    else:
        pts = outer_pts
        
    K = KDTree(pts)
              
    return (plot, pts)

def plot_outer_pts(plot, outer_pts, size = 10.0, clr = 'C0', fig_size : tuple = (18,18), altPoints = list()):
    fig, ax = plot.subplots(figsize = fig_size)
    
    X, Y = [p[0] for p in outer_pts], [p[1] for p in outer_pts]
        
    plot.scatter(X, Y, s = size, color=clr, marker = '.')
    
    return plot

# Can only handle 2d right now
def plot_CPS_2d_voronoi(plot, outer_pts, clr = list(), weights = list(), proj_region : tuple = tuple(), fig_size : tuple = (18,18), plot_points : bool = False, plot_vertices : bool = True, plot_edges : bool = True):
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        pts = pts_in_ndim_interval(proj_region, outer_pts)
    else:
        pts = outer_pts

    if len(weights) == len(outer_pts):
        vor = powerDiagram(outer_pts, weights)
    else:
        vor = Voronoi(pts)

    if plot_edges:
        fig = voronoi_plot_2d(vor, ax = ax, show_points = plot_points, show_vertices = plot_vertices, color = 'C0', marker = '.')                
    
    if len(clr) == len(outer_pts):
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            
            if (-1 not in region) and (clr[r] != 'black'):
                polygon = [vor.vertices[i] for i in region]
                plot.fill(*zip(*polygon), color = clr[r])
    
    ax.set_aspect('equal')
    
    return (plot, np.array(pts), vor)

# Can only handle 2d right now
def plot_CPS_2d_inverse_voronoi(plot, outer_pts, proj_region : tuple = tuple(), fig_size : tuple = (18,18), plot_points : bool = False, plot_vertices : bool = True):
    fig, ax = plot.subplots(figsize = fig_size)

    vor_pts = Voronoi(outer_pts).vertices
    
    if proj_region:
        pts = pts_in_ndim_interval(proj_region, vor_pts)
    else:
        pts = vor_pts
        
    vor_inverse = Voronoi(pts, qhull_options = "QJ")

    fig = voronoi_plot_2d(vor_inverse, ax = ax, show_points = plot_points, show_vertices = plot_vertices, color = 'C0', marker = '.')
    ax.set_aspect('equal')

    if plot_points:
        plot_outer_pts(plot, outer_pts, 'C1')
    
    return (plot, np.array(pts), vor_inverse)


# Can only handle 2d right now
def plot_CPS_2d_delaunay(plot,
                         outer_pts,
                         proj_region : list = [],
                         fig_size : tuple = (18,18),
                         lims : tuple = ((0,0),(0,0)),
                         out : str = '',
                         lineClr : tuple = ('C0', 1.0),
                         lineWidth : float = 1.5,
                         faceClr : tuple = ('', 0.0, 0.95),
                         altFaces : bool = False,
                         distinctFaces : bool = False,
                         printable : bool = False,
                         palette : list = list()):
    
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        pts = pts_in_ndim_interval(outer_pts, proj_region)
    else:
        pts = outer_pts

    dly_edges = list()
    dly = Delaunay(outer_pts)

    # We need to convert to a numpy array for correct indexing in the next step
    dly_verts = np.array(outer_pts)

    for splx in dly.simplices:
        tri = [list(v) for v in dly_verts[splx]]

        if ([tri[0],tri[1]] not in dly_edges) and ([tri[1],tri[0]] not in dly_edges):
            dly_edges.append([tri[0], tri[1]])
        if ([tri[1],tri[2]] not in dly_edges) and ([tri[2],tri[1]] not in dly_edges):
            dly_edges.append([tri[1], tri[2]])
        if ([tri[2],tri[0]] not in dly_edges) and ([tri[0],tri[2]] not in dly_edges):
            dly_edges.append([tri[2], tri[0]])

    fig = delaunay_plot_2d(dly, ax = ax)
        
    ax.set_aspect('equal')
    
    if not printable and not distinctFaces:
        for l in fig.axes[0].get_children():
            if type(l) is Line2D:
                l.set_color(lineClr[0])
                l.set(alpha = lineClr[1])
                l.set_linewidth(lineWidth)
                
                if faceClr[0] != '' and faceClr[1] != 0.0:
                    l.marker = None
    else:
        for l in fig.axes[0].get_children():
            if type(l) is Line2D:
                l.set(alpha = 1.0, marker = "None", linewidth = 1.0, color = "black", zorder = 5)
            
    # Color in the faces
    if faceClr[0] != '' and faceClr[1] != 0.0 and not printable:
        dly_tris = [dly.points[p] for p in dly.simplices]
          
        if altFaces:
            # Wrong but pretty
            A = lambda p : np.abs(0.5 * ((p[0][0] - p[2][0])*(p[1][1] - p[0][1]) - (p[0][0] - p[1][0])*(p[2][1] - p[2][0])))
            tri_areas = [A(p) for p in dly_tris]
        else:
            tri_areas = [shoelace_area(p) for p in dly_tris]
        
        if not distinctFaces:
            areaMax = sorted(tri_areas)[int(faceClr[2]*len(tri_areas))] #FaceClr[2] controls the intensity of the darkest tiles
        
            # Normalize the tri areas between 0.15 and 0.80
            tri_areas = [0.05 + 0.75*(t/areaMax) if t <= areaMax else 0.8 for t in tri_areas]

            if not distinctFaces:
                for i in range(len(dly_tris)):
                    plot.fill(dly_tris[i][:,0], dly_tris[i][:,1], color = faceClr[0], alpha = tri_areas[i]*faceClr[1])
        
        else:  
            """tri_areas = set(np.round(tri_areas, decimals = 2))
            area_clrs = [list(np.random.choice(np.arange(0.0, 1.0, .001), size=3)) for i in range(len(tri_areas))]
            tri_clrs = dict(zip(tri_areas, area_clrs))

            for i in range(len(dly_tris)):
                area = np.round(shoelace_area(dly_tris[i]), decimals = 2)
                clr = tri_clrs[area]
                
                plot.fill(dly_tris[i][:,0], dly_tris[i][:,1], color = clr)
                
            """
            
            similarities = [[0,],]
            
            for i in range(1, len(dly_tris)):
                tri1 = dly_tris[i]
                
                for j in range(len(similarities)):
                    triGrp = similarities[j]
                    tri2 = dly_tris[triGrp[0]]
                    
                    if i not in triGrp and tri_similar(tri1, tri2, prec = 12):
                        triGrp.append(i)
                        break
                        
                if i not in triGrp:
                    similarities.append([i,])
                    
            similarities.sort(key = lambda L : len(L), reverse = True)
                
            palette += [list(np.random.choice(np.arange(0.0, 1.0, .001), size=3)) for k in range(len(similarities) - len(palette))]
                    
            for s in range(len(similarities)):
                #clr = list(np.random.choice(np.arange(0.0, 1.0, .001), size=3))
                sim = similarities[s]
                
                for idx in sim:
                    plot.fill(dly_tris[idx][:,0], dly_tris[idx][:,1], color = palette[s])
                
                
    
    
    if len(lims) == 2 and lims != ((0,0),(0,0)):
        ax.set_xlim(lims[0][0], lims[0][1])
        ax.set_ylim(lims[1][0], lims[1][1])
    
    if out != '':
        plot.savefig(out, bbox_inches = 0, transparent=True)

    return ((plot, fig, ax), dly_verts, dly_edges, dly)

def plot_CPS_distance(plot,
                      outer_pts,
                      dist : float,
                      prec : float = 0.01,
                      dim : int = 2,
                      proj_region : list = [],
                      fig_size : tuple = (18,18),
                      plot_points : bool = True,
                      lims : tuple = ((0,0),(0,0)),
                      out : str = ''):
    
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        pts = pts_in_ndim_interval(outer_pts, proj_region)
    else:
        pts = outer_pts
    
    dist_edges = list()

    check = lambda x, y : np.isclose(np.linalg.norm(x - y), dist, atol = prec)

    for pair in combinations(outer_pts, 2):
        x, y = np.array(pair[0]), np.array(pair[1]) # Obtain the corresponding points.

        if check(x,y):
            if dim == 2:
                plot.plot([x[0],y[0]],[x[1],y[1]],color='C0')
            elif dim == 3:
                plot.plot3d([x[0],y[0]],[x[1],y[1]],[x[2],y[2]],color='C0')

            xL, yL = list(x), list(y)

            if ([xL, yL] not in dist_edges) and ([yL, xL] not in dist_edges):
                dist_edges.append([xL, yL])

    if plot_points:
        plot_outer_pts(plot, outer_pts, 'C0')
    
    if len(lims) == 2 and lims != ((0,0),(0,0)):
        ax.set_xlim(lims[0][0], lims[0][1])
        ax.set_ylim(lims[1][0], lims[1][1])
    
    if out != '':
        plot.savefig(out, bbox_inches = 0, transparent=True)
                
    return (plot, outer_pts, dist_edges)


def plot_CPS_threshold(plot,
                      outer_pts,
                      dist_bounds : tuple,
                      dim : int = 2,
                      proj_region : list = [],
                      fig_size : tuple = (-18,18),
                      plot_points : bool = True):
    
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        pts = pts_in_ndim_interval(outer_pts, proj_region)
    else:
        pts = outer_pts
    
    thresh_edges = list()

    check = lambda x, y : (dist_bounds[0] <= np.linalg.norm(x - y) <= dist_bounds[1])

    for pair in combinations(range(len(outer_pts)), 2):
        p, q = pair[0], pair[1] # We will use p and q as indices to reference the points

        x, y = pts[p], pts[q] # Obtain the corresponding points.

        if check(x,y):
            if dim == 2:
                plot.plot([x[0],y[0]],[x[1],y[1]],color='C0')
            elif dim == 3:
                plot.plot3d([x[0],y[0]],[x[1],y[1]],[x[2],y[2]],color='C0')

            xL, yL = list(x), list(y)

            if ([xL, yL] not in thresh_edges) and ([yL, xL] not in thresh_edges):
                thresh_edges.append([xL, yL])

    if plot_points:
        plot_outer_pts(plot, outer_pts, 'C0')
                
    return (plot, outer_pts, thresh_edges)

def plot_CPS_ordered_nearest_points(plot,
              outer_pts,
              order : int = 1,
              dec_places : int = 1,
              dim : int = 2,
              proj_region : list = [],
              fig_size : tuple = (18,18),
              plot_points : bool = True):
    
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        pts = pts_in_ndim_interval(outer_pts, proj_region)
    else:
        pts = outer_pts
    
    num_pts = len(outer_pts) # Quick reference for the total number of points in the projection

    # This 2D dictionary will have keys corresponding to the index of the points
    # For each key (vertex) the value will be another dictionary
    # Each sub-dictionary will have another, distinct vertex as its key
    # For each key of the sub-dictionary, the value corresponds to the distance between the two vertices
    edge_map = defaultdict(OrderedDict) 

    # Once the distance between each pair of vertices is calculated, we construct another list.
    # The first index corresponds to each vertex, V
    # The second index is a list of tuples. Each tuple contains all the other vertices which are of equal
    # distance away from V, and these tuples are ordered in ascending distance.
    dist_map = list()

    """
    STEP 1

    For each pair of points in edge_map, we calculate the distance between their projected versions
    and store this as the value corresponding to these keys.

    """
    for pair in combinations(range(num_pts),2):
        p, q = pair[0], pair[1] # We will use p and q as indices to reference the points

        x, y = np.array(pts[p]), np.array(pts[q])

        edge_map[p][q] = edge_map[q][p] = np.round(np.linalg.norm(x - y), decimals = dec_places)

    """
    STEP 2

    For each sub-dictionary in edge_map, we go through and sort these subdictionaries in ascending
    order, so that the lowest-index points correspond to the vertices that are closest.

    """
    for k in edge_map.keys():
        edge_map[k] = OrderedDict(sorted(edge_map[k].items(), key=lambda x:x[1]))

    """
    STEP 3

    We now construct dist_map using a looping method. For each vertex, this loop starts from closest
    other vertex and appends vertices of equal distance to a tuple. When it finally encounters a vertex
    that is not of equal distance, it breaks it off into a new tuple.

    """
    for k in edge_map.keys():
        neighbours = list(edge_map[k].items())

        collected_neighbours = list()

        start_dist = neighbours[0][1]
        start_list = [neighbours[0][0]]

        for pair in neighbours:
            if pair[1] <= start_dist:
                start_list.append(pair[0])
            else:
                start_dist = pair[1]
                collected_neighbours.append(start_list)
                start_list = [pair[0]]

        dist_map.append(collected_neighbours)

    """
    STEP 4

    Finally, we plot the points.    
    """
    for i in range(len(dist_map)):
        for j in range(order):

            clr = 'C0' #mcolors.hsv_to_rgb(((j*0.5)%1.0,1.0,0.9))

            for k in dist_map[i][j]:
                p = pts[i]
                q = pts[k]

                if dim == 2:
                    plot.plot([p[0],q[0]],[p[1],q[1]],color=clr)
                elif dim == 3:
                    plot.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]],color=clr)

    if plot_points:
        plot_outer_pts(plot, outer_pts, 'C0')
        
    return (plot, edge_map, dist_map)


# Requires total points (outer + inner combined)
def plot_CPS_from_edges(plot,
             total_pts,
             specified_edges : list,
             dim : int = 2,
             proj_region : list = [],
             fig_size : tuple = (-18,18),
             plot_points : bool = True,
             lims : tuple = ((0,0), (0,0)),
             out : str = ''):        
    
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        # Start by filtering out only the points with outer dimensions that lie in the region
        pts = [p for p in total_pts if in_region(p[:dim], proj_region)]
        outer_pts = [p[:dim] for p in pts]
        edges = list()
        
        # Next filter the edges whose outer dimensions lie in the region
        for e in specified_edges:
            u = e[0][:dim]
            v = e[1][:dim]
            
            if in_region(u, proj_region) and in_region(v, proj_region):
                edges.append(e)
    else:
        pts = total_pts
        outer_pts = [p[:dim] for p in total_pts]
        edges = specified_edges
    
    # Specified edges method, expects edges to be listed in terms of pairs of vertex indices. Can accept the edge output from find_lpts_in_windows
    out_edges = list()

    for e in edges:
        e_low = [e[0][:dim], e[1][:dim]]

        ex = [e_low[0][0], e_low[1][0]]
        ey = [e_low[0][1], e_low[1][1]]

        if dim == 2:
            plot.plot(ex, ey, color='C0')
            out_edges.append(((ex[0], ey[0]), (ex[1], ey[1])))
        elif dim == 3:
            ez = [e_low[0][2], e_low[1][2]]
            plot.plot3d(ex, ey, ez, color='C0')   
            out_edges.append(((ex[0], ey[0], ez[0]), (ex[1], ey[1], ez[1])))

    if plot_points:
        plot_outer_pts(plot, outer_pts, 'C0')
    
    if len(lims) == 2 and lims != ((0,0),(0,0)):
        ax.set_xlim(lims[0][0], lims[0][1])
        ax.set_ylim(lims[1][0], lims[1][1])
    
    if out != '':
        plot.savefig(out, bbox_inches = 0, transparent=True)
            
    return (plot, pts, out_edges)
    
    
def plot_CPS_recovered_edges(plot,
                             outer_pts,
                             prec : int = 4,
                             order = 0,
                             norm = 2,
                             remove_overlapping : str = '',
                             proj_region : tuple = tuple(),
                             fig_size : tuple = (-18,18),
                             lineClr = 'C0',
                             lineWidth = 1.5):
    
    fig, ax = plot.subplots(figsize = fig_size)    
    
    if proj_region:
        pts = pts_in_ndim_interval(proj_region, outer_pts)
    else:
        pts = outer_pts
        
    edges = recover_edges(pts, prec = prec, order = order, norm = norm, remove_overlapping = remove_overlapping)
    
    for pair in edges:
        dist = np.round(np.linalg.norm(pair[0]-pair[1]), decimals = 2)

        # Get the edges
        ex = [p[0] for p in pair]
        ey = [p[1] for p in pair]

        # Get the colors
        plot.plot(ex, ey, color=lineClr, linewidth = lineWidth)
        
    return (plot, pts, edges)

def plot_CPS_nearest_neighbours(plot,
                                outer_pts,
                                nn : int = 1,
                                strict = False,
                                proj_region : tuple = tuple(),
                                fig_size : tuple = (-18, 18),
                                lineClr = 'C0',
                                lineWidth = 1.5):
    
    fig, ax = plot.subplots(figsize = fig_size)     
    
    if proj_region:
        pts = pts_in_ndim_interval(proj_region, outer_pts)
    else:
        pts = np.array(outer_pts)
    
    nearestNeighbours = list()
    
    for p in pts:
        p_distances = [(q , np.linalg.norm(p - q)) for q in pts if all(p != q)]
        
        p_distances.sort(key = lambda x : x[1])
        
        if strict:
            p_nearestNeighbours = [p_distances[k][0] for k in range(nn)]
        else:
            p_nearestNeighbours = list()
            currDist = p_distances[0][1]
            
            i = 0
            j = 0
            
            while i <= len(p_distances) and j <= nn:
                if p_distances[i] == currDist:
                    p_nearestNeighbours.append(p_distances[i][0])
                    
                else:
                    currDist = p_distances[i][1]
                    j += 1
                    p_nearestNeighbours.append(p_distances[i][0])
                    
                i += 1
                
        for q in p_nearestNeighbours:
            # Get the edges
            ex = (p[0], q[0])
            ey = (p[1], q[1])

            # Get the colors
            plot.plot(ex, ey, color=lineClr, linewidth = lineWidth)
    
    return (plot, pts, nearestNeighbours)