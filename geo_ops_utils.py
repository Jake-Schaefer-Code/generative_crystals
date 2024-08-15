"""

"""
import numpy as np
from scipy.spatial import Delaunay
import itertools

def rotation_quaternions(axes, thetas):
    """

    Parameters
    ----------------
    axes

    thetas
    """
    thetas = np.atleast_1d(thetas)
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
    sin_theta = np.sin(thetas / 2)
    quaternions = np.zeros((len(thetas), 4))
    quaternions[:, 0] = np.cos(thetas / 2)
    quaternions[:, 1:] = axes * sin_theta[:, None]
    return quaternions

def make_q_rot_mats(q):
    """

    Parameters
    ----------------
    q
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot_mats = np.zeros((len(w), 3, 3))
    rot_mats[:, 0, 0] = 2 * (w * w + x * x) - 1
    rot_mats[:, 0, 1] = 2 * (x * y - w * z)
    rot_mats[:, 0, 2] = 2 * (x * z + w * y)
    rot_mats[:, 1, 0] = 2 * (x * y + w * z)
    rot_mats[:, 1, 1] = 2 * (w * w + y * y) - 1
    rot_mats[:, 1, 2] = 2 * (y * z - w * x)
    rot_mats[:, 2, 0] = 2 * (x * z - w * y)
    rot_mats[:, 2, 1] = 2 * (y * z + w * x)
    rot_mats[:, 2, 2] = 2 * (w * w + z * z) - 1
    if len(rot_mats) == 1:
        return rot_mats[0]
    return rot_mats

def q_rot_mat(theta=0, axis=np.array([0,0,1])):
    """
    Creates a rotation matrix for a given quaternion

    Parameters:
    ----------------
    q : list
        quaternion
    
    Returns:
    ----------------
    """
    q = rotation_quaternions(axis, theta)
    return make_q_rot_mats(q)

def rotate(points, theta=0, axis=np.array([0,0,1])):
    """
    Helper method. Rotates a configuration (array of positions)

    Parameters:
    ----------------
    crystal : Atoms
        The atomic structure to apply this transformation to
    theta : float, default: 0
        Angle by which to rotate the configuration
    axis : np.ndarray, default: np.array([0,0,1])
        Axis around which to rotate the configuration

    Returns:
    ----------------
    Rotated configuration
    """
    q = rotation_quaternions(axis, theta)
    rot_mat = make_q_rot_mats(q)
    return points @ rot_mat.T


def quaternion_multiply(q1, q2):
    """

    Parameters
    ----------------
    q1
    
    q2
    """
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    return np.array([w, x, y, z]).T

def generate_rotation_matrices(num_rots):
    """

    Parameters
    ----------------
    num_rots

    """
    phis, thetas = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, num_rots, endpoint=False), np.linspace(0, 2 * np.pi, num_rots, endpoint=False))
    coords = np.vstack([thetas.ravel(), phis.ravel()]).T
    axes_z = np.array([[0, 0, 1] for _ in range(len(coords[:, 0]))])
    qs_theta = rotation_quaternions(axes_z, coords[:, 0])
    matrices = make_q_rot_mats(qs_theta)
    axes_x_rot = matrices @ np.array([1,0,0])
    # axes_x_rot = np.einsum('nij,j->ni', matrices, np.array([1,0,0]))
    qs_phi = rotation_quaternions(axes_x_rot, coords[:, 1])
    qs_combined = quaternion_multiply(qs_theta, qs_phi)
    combined_rot_mats = make_q_rot_mats(qs_combined)
    return combined_rot_mats

def rotate_polyhedron(polyhedron, rotation_matrices):
    """

    Parameters
    ----------------
    polyhedron

    rotation_matrices

    """
    rotated_points = np.einsum('nij,mj->nmi', rotation_matrices, polyhedron)
    return rotated_points

def rot_mat(theta):
    """
    Parameters
    ----------------
    theta
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ])
