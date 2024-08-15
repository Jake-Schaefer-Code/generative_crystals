"""

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


from scipy.stats import gaussian_kde
from scipy.spatial import Delaunay, ConvexHull
from sklearn.mixture import GaussianMixture
from crystal_funcs import *
from TriMap import *

_cmap = plt.get_cmap('afmhot').copy()
_cmap.set_bad(color='whitesmoke')


def adjust_axis(ax, x_lim=(-1.1, 1.1), y_lim=(-1.1, 1.1)):
    """
    Parameters
    ----------------
    """
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect('equal')


def setup_axes3D(ax:Axes3D, view_angle=[30,30,0], title=None, bounds=None):
    if title is not None:
        ax.set_title(title)
    ax.view_init(*view_angle)
    ax.set_proj_type('persp')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if bounds is not None:
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        ax.set_zlim(*bounds)
    ax.set_aspect('equal')
    ax.grid(False)


def plot_polygon(vertices, ax, **kwargs):
    """
    Parameters
    ----------------
    """
    polygon = plt.Polygon(vertices[:,:2], closed=True, fill=True, edgecolor=kwargs.get('edgecolor', None), alpha=kwargs.get('alpha', 1))
    ax.add_patch(polygon)
    ax.plot(vertices[:, 0], vertices[:, 1], 'o', 
            color=kwargs.get('color', 'k'), 
            ms=4)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

def plot_polyhedron(vertices, faces):
    """
    Parameters
    ----------------
    """
    fig = plt.figure()
    ax:Axes3D = fig.add_subplot(111, projection='3d')
    poly3d = [[vertices[vertex] for vertex in face] for face in faces]
    poly_collection = Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.25)
    ax.add_collection3d(poly_collection)
    ax.scatter(*vertices.T, color='black')
    max_range = max(vertices.max(axis=0) - vertices.min(axis=0)) / 2.0
    mid = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
    bounds = (min(mid-max_range), max(mid+max_range))
    setup_axes3D(ax, bounds)
    plt.show()
    
    
def plot_transformation(polygon, transformed_polygon, ax, **kwargs):
    """
    Parameters
    ----------------
    """
    plot_polygon(polygon, ax, alpha=0.5, color='k')
    plot_polygon(transformed_polygon, ax, color='r')
    ax.quiver(polygon[:,0], polygon[:,1], 
              transformed_polygon[:,0]-polygon[:,0], transformed_polygon[:,1]-polygon[:,1], 
              angles='xy', scale_units='xy', scale=1, alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.plot(0,0,'o', color='k', ms=1)
    
def plot_simplex(simplex, ax, simplices = None, show_faces=False, **kwargs):
    """
    """
    data_kwargs = dict()
    data_kwargs["color"] = kwargs.get("color", "k")
    data_kwargs["linestyle"] = kwargs.get("linestyle", "-")
    data_kwargs["marker"] = kwargs.get("marker", "o")
    face_kwargs = dict()
    face_kwargs["s"] = kwargs.get("s", 5)
    face_kwargs["c"] = kwargs.get("c", kwargs.get("color", "k"))
    face_kwargs["alpha"] = kwargs.get("alpha", 0.05)

    faces = simplices if simplices is not None else ConvexHull(simplex).simplices
    for face in faces:
        vs = np.array(list(simplex[face]) + [simplex[face][0]])
        ax.plot(*vs.T, **data_kwargs)

        # parametrizes each face on the simplex and plots that parametrization
        # effectively shades all the faces of the simplex for better visualization
        if show_faces:
            parametrization = parametrize_triangle(*simplex[face], num_points=100)
            ax.scatter(*parametrization.T, **face_kwargs)

def plot_weighted_dist(f_sym, coords, simplex, n_samples=10000):
    """
    """
    distribution, distribution_weights = weighted_distribution(f_sym, coords, simplex, n_samples=n_samples)
    dir_points = calculate_barycentric_coordinates(simplex, distribution)
    dir_weights = barycentric_weights(dir_points)
    dir_weights /= np.sum(dir_weights)


    fig = plt.figure(figsize=(12,12))

    show_faces = False
    kwargs = {}
    if len(coords) == 3:
        subplot_kw={"projection": "3d"}
        kwargs.update(subplot_kw)
        # show_faces = True

    ax:Axes3D = fig.add_subplot(211, **kwargs)
    ax.set_title('Func on Fundamental Domain')
    scatter = ax.scatter(*distribution.T, 
            c=distribution_weights,
            alpha=0.1,
            s=10, 
            cmap=_cmap)
    fig.colorbar(scatter, ax=ax)
    plot_simplex(simplex, ax, show_faces=show_faces)

    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if len(coords) == 3:
        ax.set_zlim(0,1)
        ax.set_proj_type('persp')

    ax:Axes3D = fig.add_subplot(212, **kwargs)
    ax.set_title('Func * Dirichlet on Fundamental Domain')
    scatter = ax.scatter(*distribution.T, 
            c=distribution_weights*dir_weights,
            alpha=0.1,
            s=10, 
            cmap=_cmap)
    fig.colorbar(scatter, ax=ax)    
    plot_simplex(simplex, ax, show_faces=show_faces)

    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if len(coords) == 3:
        ax.set_zlim(0,1)
        ax.set_proj_type('persp')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# TODO plot contours of 90% intervals?
def plot_func(func, *xi, center, sym_ops):
    """
    """
    # sym_ops = generate_space_group(symbol)
    # plot_names = ['Original', f'Symmetrized ({symbol})']
    # for ax, group, title in zip(axes, data_groups, plot_names):
    #     ax.set_title(title)

    data_groups = [apply_func(func, *xi)]
    for op in sym_ops:
        data_groups.append(symmetrize(func, *xi, sym_ops=op))
    
    kwargs = {}
    if len(xi) == 3:
        subplot_kw={"projection": "3d"}
        kwargs.update(subplot_kw)
    
    fig = plt.figure(figsize=(12,16))
    axes = [fig.add_subplot(len(data_groups), 1, i+1, **kwargs) for i in range(len(data_groups))]    
    for ax, group in zip(axes, data_groups):
        coords = [(x+center[i]).ravel() for i, x in enumerate(xi)]
        values = group.ravel()
        scatter = ax.scatter(*coords, c=values, cmap=_cmap, alpha=0.5)
        fig.colorbar(scatter, ax=ax)
        # ax.set_proj_type('persp')
        ax.set_aspect('equal')
    plt.show()




def plot_contours(X, Y, center, data, fund_unit, ax):
    ax.contourf(X + center[0], Y + center[1], data, levels=100)
    patch = plt.Polygon(fund_unit, fill=False, edgecolor='k')
    ax.add_patch(patch)

def plot_density(coords, weights, fund_unit, ax):
    kde = gaussian_kde([*coords.T], weights=weights)
    X_grid, Y_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z_grid = kde([X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)
    cf = ax.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap=_cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cf, cax=cax, orientation='vertical')
    patch = plt.Polygon(fund_unit, fill=False, edgecolor='k')
    ax.add_patch(patch)

def plot_symmetry_op(X, Y, domain, group_data, fund_unit, dirichlet_pts, dirichlet_weights, n_samples, group_name, axes):
    """
    """
    center = 0.5 * (domain[:,1] - domain[:,0])
    coords = np.vstack([X.ravel(), Y.ravel()]).T + center
    cart_pts = barycentric_to_cartesian_2D(dirichlet_pts, fund_unit)

    
    inside_coords = coords[isinside(coords, fund_unit)]
    weights = distance_weights(inside_coords, cart_pts, w=dirichlet_weights)
    n_comp = 4 if group_name == 'p4gm' else 1
    gmm = GaussianMixture(n_components=n_comp, max_iter=1000)
    probabilities = group_data.ravel()**2
    probabilities /= np.sum(probabilities)
    dist = np.random.choice(len(coords), size=n_samples, p=probabilities)
    gmm.fit(coords[dist])
    samples, _ = gmm.sample(n_samples)
    inside_samples = samples[isinside(samples, fund_unit)]
    inside_weights = distance_weights(inside_coords, inside_samples)
    total_weights = weights * inside_weights
    total_weights /= np.sum(total_weights)    
    filtered_coords = inside_coords[total_weights > 0]
    filtered_weights = total_weights[total_weights > 0]
    plot_contours(X, Y, center, group_data, fund_unit, axes[0])
    plot_density(filtered_coords, filtered_weights, fund_unit, axes[1])
    for ax in axes:
        ax.set_title(group_name)
        adjust_axis(ax, x_lim=(0,0.99),y_lim=(0,0.99))


def plot_result(map:TriMap, idx:int):
    best_map = map.maps[idx]
    best_poly_oriented = map.rotated_polygons[idx]
    best_matrices = map.matrices[idx]
    inv_mats = np.array([np.linalg.inv(M) for M in best_matrices])
    dist = barycentric_to_cartesian_2D(map.distribution, np.roll(map.triangle,-1,axis=0))
    dst_idxs = get_triangles(best_map, dist)
    trans_dst = np.array([apply_transformation(pt, inv_mats[idx]) for pt, idx in zip(dist, dst_idxs)])
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    plot_transformation(best_poly_oriented, best_map, ax[0])
    # plot_transformation(best_map, best_poly_oriented, ax[1])
    best_tri = map.triangulations[idx]
    for t in best_tri.simplices:
        plot_polygon(best_poly_oriented[t], ax[1], alpha=0.1, edgecolor='k')

    ax[0].scatter(dist[:, 0], 
                  dist[:, 1], 
                  s=10, 
                  c=map.distribution_weights, 
                  cmap=_cmap, 
                  alpha=0.5)
    ax[1].scatter(trans_dst[:, 0], 
                  trans_dst[:, 1], 
                  s=10, 
                  c=map.distribution_weights, 
                  cmap=_cmap, 
                  alpha=0.5)
    plt.show()