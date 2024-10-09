import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

def plot_outer_pts(plot, outer_pts, clr = 'C0', fig_size : tuple = (18,18)):
    fig, ax = plot.subplots(figsize = fig_size)
    
    X, Y = [p[0] for p in outer_pts], [p[1] for p in outer_pts]
        
    plot.scatter(X, Y, color=clr, marker = '.')
    
    return plot

# Can only handle 2d right now
def plot_CPS_2d_voronoi(plot, outer_pts, proj_region : tuple = tuple(), fig_size : tuple = (18,18), plot_points : bool = False, plot_vertices : bool = True):
    fig, ax = plot.subplots(figsize = fig_size)
    
    if proj_region:
        pts = pts_in_ndim_interval(proj_region, outer_pts)
    else:
        pts = outer_pts

    vor = Voronoi(pts)

    fig = voronoi_plot_2d(vor, ax = ax, show_points = plot_points, show_vertices = plot_vertices, color = 'C0', marker = '.')
    ax.set_aspect('equal')

    if plot_points:
        plot_outer_pts(plot, outer_pts, 'C1')
    
    return (plot, np.array(pts), vor)