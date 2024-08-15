"""

"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
import itertools
from geo_ops_utils import *

def add_points_to_edge(v1, v2, num_pts=2):
    """
    Parameters
    ----------------
    v1
    
    v2
    
    num_pts=2
    """
    # excludes first point bc vertex
    return [(1 - t) * v1 + t * v2 for t in np.linspace(0, 1, num_pts+1, endpoint=False)[1:]]

def edge_func(v1, v2, num_pts=20):
    """

    Parameters
    ----------------
    
    """
    t = np.linspace(0,1,num_pts)[1:]
    return t[:, None] * v1 + (1-t[:, None]) * v2

def add_points(polygon, pts_per_edge=2):
    """
    Creates points on the edges of the polygon, distributed evnlt 
    again, polygon vertices n eed to be ordered counterclockwise

    Parameters
    ----------------
    polygon
    
    pts_per_edge=2
    """
    num_vrt = len(polygon)
    # return np.array([*add_points_to_edge(polygon[i], polygon[(i + 1) % num_vrt]) for i in range(len(polygon))])
    return np.array([(1 - t) * polygon[i] + t * polygon[(i + 1) % num_vrt] 
                         for i in range(num_vrt)
                         for t in np.linspace(0, 1, pts_per_edge, endpoint=False)])

def get_triangles(polygon, pts):
    """
    Parameters
    ----------------
    polygon
    
    pts
    """
    triangulation = Delaunay(polygon[:,:2])
    return triangulation.find_simplex(pts)

def get_triangles2(triangulation, pts):
    """

    Parameters
    ----------------
    
    """
    truths = np.array([isinside(pts, triangle) for triangle in triangulation])
    result = np.argmax(truths, axis=0)
    return result


def is_inside_tetrahedron(points, tetrahedron):
    """

    Parameters
    ----------------
    
    """
    """TODO do this with barycentric coords"""
    hull = Delaunay(tetrahedron)
    return hull.find_simplex(points) >= 0

def tri_area(triangles):
    """
    Parameters
    ----------------
    triangles
    """
    return 0.5 * (triangles[..., 0, 0] * (triangles[..., 1, 1] - triangles[..., 2, 1]) + 
                  triangles[..., 1, 0] * (triangles[..., 2, 1] - triangles[..., 0, 1]) +
                  triangles[..., 2, 0] * (triangles[..., 0, 1] - triangles[..., 1, 1]))

def area(vertices):
    """
    vetices must be ordered ccw
    """
    return 0.5 * np.sum(vertices[..., 0] * np.roll(vertices[..., 1], 1, axis=-1) -
                        vertices[..., 1] * np.roll(vertices[..., 0], 1, axis=-1), axis=-1)

def unsigned_area(vertices):
    """
    vetices must be ordered ccw
    """
    return 0.5 * np.abs(np.sum(vertices[..., 0] * np.roll(vertices[..., 1], 1, axis=-1) -
                               vertices[..., 1] * np.roll(vertices[..., 0], 1, axis=-1), axis=-1))
 
def isinside(points, triangle):
    """
    Parameters
    ----------------
    points:np.ndarray
    
    triangle:np.ndarray
    """
    a1 = unsigned_area(triangle)
    tri = np.full((len(points),3,2), triangle)
    tri[:,0] = points
    a2 = unsigned_area(tri)
    tri = np.full((len(points),3,2), triangle)
    tri[:,1] = points
    a3 = unsigned_area(tri)
    tri = np.full((len(points),3,2), triangle)
    tri[:,2] = points
    a4 = unsigned_area(tri)
    return (a1 == a2 + a3 + a4)

def barycentric_coordinates2D(p, triangle):    
    """
    Parameters
    ----------------
    p
    
    a
    
    b
    
    c
    """
    a,b,c = triangle[0],triangle[1],triangle[2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0.T)
    d21 = np.dot(v2, v1.T)
    denom = d00 * d11 - d01 * d01
    l1 = (d11 * d20 - d01 * d21) / denom
    l2 = (d00 * d21 - d01 * d20) / denom
    l3 = 1-l1-l2
    return np.vstack([l1, l2, l3]).T

def barycentric_to_cartesian_2D(p, vertices):
    """
    vertices must be 3 vertices in cartesian
    """
    return np.dot(p, vertices)

def cart_to_bary3D(points, vertices):
    """

    Parameters
    ----------------
    
    """
    a,b,c,d = vertices
    T = np.vstack([b - a, c - a, d - a]).T
    T_inv = np.linalg.inv(T)
    bary_coords = np.dot(points - a, T_inv)
    # adding l4
    # TODO axis was originally set to axis=1, but changed to axis=-1. Confirm this
    bary_coords = np.hstack((1 - bary_coords.sum(axis=-1, keepdims=True), bary_coords))
    return bary_coords

def bary_to_cart(points, vertices):
    """

    Parameters
    ----------------
    points

    vertices
    """
    return sum([points[:, i][:, np.newaxis]*v for i, v in enumerate(vertices)])

def barycentric_weights(point:np.ndarray):
    """
    Weights based on product of barycentric coordinates: 
    if along edge, one of the coordinates == 0, 
    so distribution value will be 0

    Parameters
    ----------------
    point:np.ndarray
    """
    return np.prod(point, axis=-1)

def calculate_barycentric_coordinates(vertices, points):
    """
    Parameters
    ----------------
    vertices

    points
    """
    A = np.vstack([vertices.T, np.ones(vertices.shape[0])])
    b = np.hstack([points, np.ones((points.shape[0],1))]).T
    barycentric_coords, r, _, _ = np.linalg.lstsq(A, b, rcond=None)
    if np.any(r):
        print(f"barycentric residual: {r}")
    return barycentric_coords.T


def centroid(polygon):
    """
    Parameters
    ----------------
    polygon
    """
    return np.array([np.sum(polygon[:,0])/len(polygon[:,0]), np.sum(polygon[:,1])/len(polygon[:,1])])

def tetrahedron_volume(tetrahedrons):
    """

    Parameters
    ----------------
    
    """
    return np.abs(np.sum(
        np.cross(tetrahedrons[..., 1, :] - tetrahedrons[..., 0, :], 
                 tetrahedrons[..., 2, :] - tetrahedrons[..., 0, :]) * 
                 (tetrahedrons[..., 3, :] - tetrahedrons[..., 0, :]), axis=-1)) / 6

def simplex_centroid(simplices):
    """

    Parameters
    ----------------
    
    """
    return np.sum(simplices, axis=-2)/simplices.shape[-2]

def polyhedron_centroid(vertices):
    """

    Parameters
    ----------------
    
    """
    simplices = vertices[Delaunay(vertices).simplices]
    volumes = tetrahedron_volume(simplices)
    centroids = simplex_centroid(simplices)
    return np.sum(centroids * volumes[:,np.newaxis], axis=0) / np.sum(volumes)

def unique_triples(polygon):
    """
    Parameters
    ----------------
    polygon
    """
    indices = np.arange(0, len(polygon))
    x, y, z = np.meshgrid(indices, indices, indices)
    triples_indices = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    triples_indices = triples_indices[((triples_indices[:, 0] != triples_indices[:, 1]) & 
                                       (triples_indices[:, 1] != triples_indices[:, 2]) &
                                       (triples_indices[:, 2] != triples_indices[:, 0]))]
    triples_indices = np.unique(np.sort(triples_indices, axis=1), axis=0)
    return triples_indices

def unique_tuples(num_pts, tuple_len):
    """
    Parameters
    ----------------
    num_pts, tuple_len
    """
    indices = np.arange(0, num_pts)
    combinations = np.array(np.meshgrid(*[indices for _ in range(tuple_len)])).T.reshape(-1, tuple_len)
    combinations.sort(axis=1)
    unique_mask = np.all(np.diff(combinations, axis=1) != 0, axis=1)
    return np.unique(combinations[unique_mask], axis=0)

def orientation(p1, p2, p3):
    """
    Parameters
    ----------------
    p1
    
    p2
    
    p3
    """
    val = np.cross(p1[:2]-p2[:2], p3[:2]-p2[:2])
    # val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2
    
def on_segment(p, q, r):
    """
    Parameters
    ----------------
    """
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and 
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

def segments_intersect(p1, q1, p2, q2):
    """
    Parameters
    ----------------
    p1
    
    q1
    
    p2
    
    q2
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    return ((o1 != o2 and o3 != o4) or
            (o1 == 0 and on_segment(p1, p2, q1)) or
            (o2 == 0 and on_segment(p1, q2, q1)) or
            (o3 == 0 and on_segment(p2, p1, q2)) or
            (o4 == 0 and on_segment(p2, q1, q2)))

def is_self_intersecting(polygon):
    """
    Parameters
    ----------------
    polygon
    """
    n = len(polygon)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return True
    return False


def dist_to_line(points, lines_start, lines_end):
    """

    Parameters
    ----------------
    
    """
    line_vectors = lines_end - lines_start
    line_vectors_ext = line_vectors[:, np.newaxis, :]
    point_vectors = points - lines_start[:, np.newaxis, :] 
    line_lengths_squared = np.sum(line_vectors**2, axis=1)  
    line_lengths_squared[line_lengths_squared == 0] = np.inf  
    line_lengths_squared_ext = line_lengths_squared[:, np.newaxis]  
    t = np.sum(point_vectors * line_vectors_ext, axis=2) / line_lengths_squared_ext  
    t = np.clip(t, 0, 1)
    nearest_points = lines_start[:, np.newaxis, :] + t[:, :, np.newaxis] * line_vectors_ext 
    return np.linalg.norm(point_vectors - nearest_points, axis=2) 


# Mapping and Transformations

def get_parameter(p, v1, v2, t1, t2, kind):
    """
    Parameters
    ----------------
    p
    
    v1
    
    v2
    """

    vec = v2-v1
    dist_p_v1 = np.linalg.norm(p - v1)
    dist_p_v2 = np.linalg.norm(p - v2)
    if kind == 0:
        param = (dist_p_v1 / (dist_p_v1 + dist_p_v2))
    elif kind == 1:
        param = np.dot(p-v1, vec) / np.dot(vec,vec)
    elif kind == 2:
        area_a = np.linalg.norm(np.cross(p-v1, p - t1)) 
        area_b = np.linalg.norm(np.cross(p-v2 , p - t2)) 
        param = (area_a / (area_a + area_b)) * (dist_p_v1 / (dist_p_v1 + dist_p_v2))
    return np.clip(param, 0.01, 0.99)

def interpolate_vertex(p, v1, v2, t1, t2, kind):    
    """
    Parameters
    ----------------
    p
    
    v1
    
    v2
    
    t1
    
    t2
    """
    param = get_parameter(p, v1, v2, t1, t2, kind)
    return t1 + param * (t2-t1)

def interpolate_to_plane(p, t1, t2, t3, v1, v2, v3):
    """

    Parameters
    ----------------
    
    """
    dist_p_v1 = np.linalg.norm(p - v1)
    dist_p_v2 = np.linalg.norm(p - v2)
    dist_p_v3 = np.linalg.norm(p - v3)
    param1 = (dist_p_v1 / (dist_p_v1 + dist_p_v2))
    param2 = (dist_p_v2 / (dist_p_v2 + dist_p_v3))
    return ((t1 + param1 * (t2-t1)) + (t2 + param2 * (t3-t2)))/2

def slerp(v0, v1, t):
    """

    Parameters
    ----------------
    
    """
    dot = np.dot(v0, v1)
    theta = np.arccos(dot) * t
    sin_theta_0 = np.sin((1 - t) * theta)
    sin_theta_1 = np.sin(t * theta)
    return  (sin_theta_0 * v0 + sin_theta_1 * v1) / np.sin(theta)

def orthogonal_complement(q1, q2, q3):
    """

    Parameters
    ----------------
    
    """
    matrix = np.array([q1, q2, q3])
    u, s, vh = np.linalg.svd(matrix)
    return vh[-1]

# instead of doing this, rotate orthogonal to q
def orthogonal_projection(p, q):
    """
    projects p perpendicular to q

    Parameters
    ----------------

    """
    # projection = p - (np.dot(p,q)/np.dot(q,q)*q)
    projection = p - (np.dot(p,q)*q)
    return projection/np.linalg.norm(projection)



def polynomial_basis(x, y, max_degree):
    """
    Parameters
    ----------------
    x
    
    y
    
    max_degree
    """
    return np.array([(x**i) * (y**j) for i in range(max_degree + 1) for j in range(max_degree + 1 - i)])

def polynomial_basis3D(x, y, z, max_degree):
    """

    Parameters
    ----------------
    
    """
    return np.array([(x**i) * (y**j) * (z**k)
                     for i in range(max_degree + 1) 
                     for j in range(max_degree + 1 - i) 
                     for k in range(max_degree + 1 - i - j)])

def polynomial_basis4D(x, y, z,w, max_degree):
    """

    Parameters
    ----------------
    
    """
    return np.array([(x**i) * (y**j) * (z**k) * (w**l)
                     for i in range(max_degree + 1) 
                     for j in range(max_degree + 1 - i) 
                     for k in range(max_degree + 1 - i - j)
                     for l in range(max_degree + 1 - i -j - k)])

def compute_coefficients(points_src, points_dst, max_degree):
    """
    Parameters
    ----------------
    points_src
    
    points_dst
    
    max_degree
    """
    if len(points_src) != len(points_dst):
        raise IndexError(f"Must have the same amount of source points and destination points: {len(points_src)} != {len(points_dst)}")
    A = np.zeros((len(points_src), (max_degree + 1) * (max_degree + 2) // 2))
    for i, (x, y) in enumerate(points_src):
        A[i, :] = polynomial_basis(x, y, max_degree)
    coeffs_x, _, _, _ = np.linalg.lstsq(A, points_dst[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, points_dst[:, 1], rcond=None)
    return coeffs_x, coeffs_y



def compute_coefficients3D(points_src, points_dst, max_degree):
    """

    Parameters
    ----------------
    
    """
    if len(points_src) != len(points_dst):
        raise IndexError(f"Must have the same amount of source points and destination points: {len(points_src)} != {len(points_dst)}")
    dim = points_src.shape[1]
    
    A = np.zeros((len(points_src), int(np.prod([(max_degree+1+i)/(1+i) for i in range(dim)]))))
    for i, (x, y, z) in enumerate(points_src):
        A[i, :] = polynomial_basis3D(x, y, z, max_degree)
    coeffs = [np.linalg.lstsq(A, points_dst[:, i], rcond=None)[0] for i in range(dim)]
    return coeffs


def transformation_matrix(points_src, points_dst):
    """
    Parameters
    ----------------
    points_src
    
    points_dst
    """
    A = np.vstack([points_src.T, np.ones((points_src.shape[0],1)).T]).T
    B = np.vstack([points_dst.T, np.ones((points_dst.shape[0],1)).T]).T
    transformation_matrix, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return transformation_matrix.T

def apply_transformation(point, matrix):
    """
    Parameters
    ----------------
    point
    
    matrix
    """

    return(matrix @ np.append(point, 1))[:-1]


def create_piecewise_matrices(points_src, points_dst, triangle_indices):
    """
    Parameters
    ----------------
    points_src
    
    points_dst
    """
    return np.array([transformation_matrix(points_src[s2], points_dst[s2]) for s2 in triangle_indices])



def to_sphere(points):
    """

    Parameters
    ----------------
    
    """
    points = np.hstack((points, -np.ones((points.shape[0], 1))))
    return points / np.linalg.norm(points, axis=1, keepdims=True)

def from_sphere(points):
    """

    Parameters
    ----------------
    
    """
    scale_factor = -1/points[:,-1]
    return points[:,:-1] * scale_factor[:,None]

def stereographic_projection(points):
    """

    Parameters
    ----------------
    
    """
    # denom = points[:,-1, None]+1
    denom = 1-points[:,-1, None]
    denom = np.where(denom==0, 1, denom)
    return 2*points[:,:-1] / denom

def inv_stereo_proj(*xi):
    """

    Parameters
    ----------------
    
    """
    denom = 4 + np.sum([x**2 for x in xi], axis=0)
    proj_coords = np.vstack([(4*x)/denom for x in xi]+[(denom-8)/denom]).T
    return proj_coords

def cart_to_sph(coords):
    """

    Parameters
    ----------------
    
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.vstack((theta, phi)).T

def sph_to_cart(coords):
    """

    Parameters
    ----------------
    
    """
    theta = coords[:, 0]
    phi = coords[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z)).T
    # return np.column_stack((x, y, z))

def geodesic(p1, p2, num_points=100):
    """

    Parameters
    ----------------
    
    """
    p1 /= np.linalg.norm(p1)
    p2 /= np.linalg.norm(p2)
    t = np.linspace(0, 1, num_points)[:, None]
    theta = np.arccos(np.dot(p1, p2))
    sin_theta = np.sin(theta)
    if sin_theta==0: 
        print(f"geodesic(): divide by zero: sin(theta)=0, cos(theta)={np.dot(p1, p2)}")
        sin_theta=1
    
    return (np.sin((1 - t) * theta) / sin_theta) * p1 + (np.sin(t * theta) / sin_theta) * p2

def dist_to_geodesic(p,v1,v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    p /= np.linalg.norm(p)
    n = np.cross(v1,v2)
    n /= np.linalg.norm(n)
    theta = np.arccos(np.dot(p,n))
    return np.pi/2-theta 

def rotate_around_axis(v, k, theta):
    """

    Parameters
    ----------------

    """
    k = k / np.linalg.norm(k) 
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    v_rot = v * cos_theta + np.cross(k, v) * sin_theta + k * np.dot(k, v) * (1 - cos_theta)
    return v_rot

def get_smallest_angles(polygon):
    """
    3 vertices with smallest angles in poly

    Parameters
    ----------------
    
    """
    num_vertices = polygon.shape[0]
    angles = np.zeros(num_vertices)
    for i in range(num_vertices):
        v1 = polygon[(i - 1) % num_vertices] - polygon[i]
        v2 = polygon[(i + 1) % num_vertices] - polygon[i]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angles[i] = np.arccos(np.dot(v1, v2))
    return np.argsort(angles)

def find_vertex_angles(polyhedron, adjacency_list):
    """
    Parameters
    ----------------
    polyhedron

    adjacency_list
    """
    angles = np.zeros(len(polyhedron))
    for i, v in enumerate(polyhedron):
        edge_vecs = polyhedron[adjacency_list[i]] - v[None, :]
        edge_vecs /= np.linalg.norm(edge_vecs, axis=1, keepdims=True)
        pairs = list(itertools.combinations(range(len(edge_vecs)), 2))
        thetas = np.arccos(np.sum(edge_vecs[pairs][:,0] * edge_vecs[pairs][:,1], axis=-1))
        angles[i] = np.sum(thetas)
    return np.argsort(angles)


def parametrize_geodesic_triangle(q1,q2,q3, num_points=10):
    """
    Parameters
    ----------------
    q1
    
    q2
    
    q3
    
    num_points=10
    """
    s, t = np.meshgrid(np.linspace(0,1,num_points+1), np.linspace(0,1,num_points+1))
    mask = (s + t <= 1)
    s, t = s[mask][:,None], t[mask][:,None]
    points = s * q1 + t * q2 + (1-s-t)*q3
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def parametrize_triangle(p1,p2,p3, num_points=10):
    """
    Parameters
    ----------------
    p1
    
    p2
    
    p3
    
    num_points=10
    """
    alpha, beta = np.meshgrid(np.linspace(0,1,num_points+1), np.linspace(0,1,num_points+1))
    alpha = alpha.ravel()
    beta = beta.ravel()
    mask = (alpha + beta <= 1)
    alpha, beta = alpha[mask], beta[mask]
    gamma = 1 - alpha - beta
    points = alpha[:, np.newaxis] * np.array(p1) + \
             beta[:, np.newaxis] * np.array(p2) + \
             gamma[:, np.newaxis] * np.array(p3)
    return points

def scale_poly(points):
    """
    Parameters
    ----------------
    points
    """
    return points / np.max(np.linalg.norm(points, axis=1))


def dirichlet_energy(matrices, mapping, triangulation, polyhedron):
    """
    Parameters
    ----------------
    matrices:np.ndarray, 
    
    mapping:np.ndarray,
    
    triangulation:np.ndarray
    
    polyhedron
    
    """
    # TODO dont need to calculate volumes every time (in class)

    volumesA = tetrahedron_volume(polyhedron[triangulation])
    volumesB = tetrahedron_volume(mapping[triangulation])
    # volumesA = np.where(volumesA==0, 1e-16, volumesA)
    if (volumesA == 0).any() or (volumesB==0).any():
            return np.inf
    try:
        # B * len(triangulation) / np.sum(volumesB)
        spectral_norms = np.array([max(np.linalg.svd(J)[1])**2*B + max(np.linalg.svd(np.linalg.inv(J))[1])**2*A
                                for J, A, B in zip(matrices, volumesA, volumesB)])
        return np.sum(spectral_norms)
    except:
        print("singular matrices")
        return np.inf

def minimize_jacobian(matrices, maps, triangulations, rotated_polygons):
    """
    Parameters
    ----------------
    matrices, 
    
    maps, 
    
    triangulations, 
    
    rotated_polygons
        
    """
    energies = np.array([dirichlet_energy(mats, maps, tris, polys) 
                            for mats, maps, tris, polys in zip(matrices, maps, triangulations, rotated_polygons)])
    print(energies[np.argmin(energies)])
    return np.argmin(energies)
def apply_func(func, *xi, domain=None):
    """
    Parameters
    ----------------
    func, 

    *xi, 
    
    domain=None
    """
    if domain is None: domain = np.array([(0,1) for _ in range(len(xi))])
    center = 0.5 * (domain[:,1] - domain[:,0])
    # modulo [0,1) domain, assuming that the coords were translated to be centred at zero
    xp = [(x + center[i]) % (2*center[i]) for i, x in enumerate(xi)]
    return func(*xp)

def symmetrize(func, *xi, sym_ops, domain=None):
    """
    Parameters
    ----------------
    func, 
    
    *xi, 
    
    sym_ops, 
    
    domain=None
    """
    f_symm = np.zeros_like(xi[0])
    for M, t in sym_ops:
        X_prime = np.dot(M, np.vstack([x.ravel() for x in xi])) + t[:, np.newaxis]
        xp = [X_prime[i].reshape(x.shape) for i, x in enumerate(xi)]
        f_symm += apply_func(func, *xp, domain=domain)
    return f_symm/len(sym_ops)


def reflection_matrix(normal):
    """
    R = I - 2nn^T
    
    Parameters:
    ----------------
    normal : arraylike
        The normal vector to the plane.
        
    Returns:
    --------
    R : np.ndarray, shape = (ndim, ndim)
    """
    normal = np.array(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    I = np.eye(len(normal))
    return I - 2 * np.outer(normal, normal)

def glide_reflection(normal, translation):
    """
    """
    return (reflection_matrix(normal), translation)

def screw_rotation(theta, axis, translation):
    """
    """
    return (q_rot_mat(theta, axis), translation*axis)

def identity(inversion=False):
    """
    """
    if inversion:
        return (-np.eye(3), np.zeros(3))
    return (np.eye(3), np.zeros(3))

def interpolate_to_plane(p, t1, t2, t3, v1, v2, v3):
    """
    """
    dist_p_v1 = np.linalg.norm(p - v1, axis=1, keepdims=True)
    dist_p_v2 = np.linalg.norm(p - v2, axis=1, keepdims=True)
    dist_p_v3 = np.linalg.norm(p - v3, axis=1, keepdims=True)
    param1 = (dist_p_v1 / (dist_p_v1 + dist_p_v2))
    param2 = (dist_p_v2 / (dist_p_v2 + dist_p_v3))
    return ((t1 + param1 * (t2-t1)) + (t2 + param2 * (t3-t2)))/2

def parse_rotation(symbol, i, axis):
    """
    """
    n = int(symbol[i])
    return [(q_rot_mat(j * 2 * np.pi / n, axis), np.zeros(3)) for j in range(n)]

def parse_mirror_plane(axis):
    """
    """
    return (reflection_matrix(axis), np.zeros(3))

def parse_glide_plane(symbol, i):
    """
    """
    if symbol[i - 1] == 'a':
        axis = np.array([1., 0, 0])
    elif symbol[i - 1] == 'b':
        axis = np.array([0, 1., 0])
    elif symbol[i - 1] == 'c':
        axis = np.array([0, 0, 1.])
    else:
        raise NameError(f"Unknown axis at position {i}: {symbol[i]}")
    return glide_reflection(axis, 0.5*axis)

def parse_screw_axis(symbol, i, axis_z):
    """
    """
    n = int(symbol[i])
    return [screw_rotation(2 * np.pi / n, axis_z, np.array([0, 0, 0.5]))]


def parse_space_group(symbol):
    """
    """
    sym_ops = [identity()]
    axis_z = np.array([0, 0, 1.])
    axis_x = np.array([1., 0, 0])
    axis_y = np.array([0, 1., 0])

    i = 0
    while i < len(symbol):
        char = symbol[i]
        if char.isdigit() and (i > 0 and symbol[i-1].lower() == 'p'):
                sym_ops.extend(parse_rotation(symbol, i, axis_z))
        
        elif char == 'm': 
            if i > 0 and symbol[i - 1].isdigit():  
                sym_ops.append(parse_mirror_plane(axis_z))
                if i + 1 < len(symbol) and symbol[i + 1] == 'm':
                    sym_ops.append(parse_mirror_plane(axis_x))  
                    sym_ops.append(parse_mirror_plane(axis_y))  
            else:  
                sym_ops.append(parse_mirror_plane(axis_z))

        elif char == 'n': 
            sym_ops.append(parse_glide_plane(symbol, i))
        
        elif char == '/':  
            i += 1
            if i < len(symbol) and symbol[i].isdigit():
                sym_ops.extend(parse_screw_axis(symbol, i, axis_z))
            elif i < len(symbol) and symbol[i] == 'm':
                sym_ops.append(parse_mirror_plane(axis_z))  
            elif i < len(symbol) and symbol[i] == 'c':
                sym_ops.append(glide_reflection(axis_z, np.array([0, 0.5, 0.5])))  
    
        elif char == '1':  
            sym_ops.append(identity())
        
        # TODO more
        
        i += 1
    return sym_ops

def print_symmetry_operations(sym_ops):
    for op in sym_ops:
        print("Rotation/Reflection Matrix:\n", op[0])
        print("Translation Vector:\n", op[1])
        print("-" * 30)


def distance_weights(coords, points, w=None):
    """
    weights based on distances from points
    """
    distances = np.linalg.norm(coords[:, np.newaxis, :] - points, axis=2)
    closest_indices = np.argmin(distances, axis=0)
    weight_sums = np.bincount(closest_indices, minlength=coords.shape[0])
    if w is None:
        return weight_sums / np.sum(weight_sums)
    weights = np.bincount(closest_indices, weights=w, minlength=coords.shape[0])
    # mean weight value at each coordinate
    weights /= weight_sums
    # norm to 1
    return weights / np.sum(weights)

def weighted_distribution(f_sym, coordinates, simplex, alpha=None, n_samples=10000):
    """
    coordinates must be of shape (ndim, npts)
    """
    if alpha is None: alpha = np.ones(len(simplex))

    dir_points = np.random.dirichlet(alpha, n_samples)
    dir_points_cart = bary_to_cart(dir_points, simplex)
    dir_weights = barycentric_weights(dir_points)
    dir_weights /= np.sum(dir_weights)
    f_weights = f_sym.ravel() / np.sum(f_sym.ravel())
    kde = gaussian_kde([*coordinates], weights=f_weights, bw_method=0.05)
    sampled_f = kde(dir_points_cart.T)
    sampled_f /= np.sum(sampled_f)

    f_values_min = np.min(sampled_f)
    f_values_max = np.max(sampled_f)
    f_sampled_normalized = (sampled_f - f_values_min) / (f_values_max - f_values_min)
    f_sampled_normalized /= np.sum(f_sampled_normalized)
    probabilities = f_sampled_normalized #* dir_weights
    return dir_points_cart, probabilities


def collect_matrices(src_tetra:np.ndarray, dst_ply:np.ndarray, num_rots=12):
    """
    Ex: src_ply is the fund domain simplex with defined distribution, 
    dst_ply is the destination polytope
    """
    dst_w_cent = np.array(list(dst_ply) + [polyhedron_centroid(dst_ply)])

    # TODO look at this more - something is up with the way it isnt covering the top of the sphere
    mats = generate_rotation_matrices(num_rots)
    rotated_plys = rotate_polyhedron(dst_w_cent, mats)
    combos = unique_tuples(dst_ply.shape[0], 4)

    piecewise_matrices, dst, triangulations, src, simps = [], [], [], [], []
    for combo in combos:
        rotated_tetras = rotated_plys[:, combo, :]
        mask = np.ones(dst_w_cent.shape[0], dtype=bool)
        mask[combo] = False
        mask[-1] = False
        # other pts
        others = rotated_plys[:, mask, :]
        dst_pts = np.zeros_like(rotated_plys)
        dst_pts[:, combo, :] = src_tetra
        
        # bary coords for other pts
        bary_others = np.array([calculate_barycentric_coordinates(src_tetra, rot) for rot in others])
        # Should I do abs. val of 
        closest_face = np.argsort(bary_others, axis=-1)[:,:, -3:]
        
        rows = np.arange(others.shape[0])[:, None, None]
        faces = src_tetra[closest_face]
        faces_poly = rotated_tetras[rows, closest_face]
        interped_pts = np.array([
            interpolate_to_plane(pts, tetra_face[:,0], tetra_face[:,1], tetra_face[:,2], 
                                 poly_face[:, 0], poly_face[:, 1], poly_face[:, 2]) 
                                 for pts, tetra_face, poly_face in zip(others, faces, faces_poly)
                                 ])
        
        dst_pts[:, mask, :] = interped_pts

        tri_dst = [Delaunay(pt) for pt in dst_pts]
        simplices = [tri.simplices for tri in tri_dst]

        triangulations.extend(tri_dst)
        simps.extend(simplices)
        dst.extend(list(dst_pts.copy()))
        src.extend(list(rotated_plys.copy()))

        piecewise_matrices.extend([
            create_piecewise_matrices(src_pt, dst_pt, tri) 
            for src_pt, dst_pt, tri in zip(rotated_plys, dst_pts, simplices)
            ])
        
    return piecewise_matrices, triangulations, simps, dst, src