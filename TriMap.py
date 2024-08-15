import numpy as np
from scipy.spatial import Delaunay
from crystal_funcs import *
from plotting_utils import *

class TriMap:
    """
    """
    def __init__(self, polygon:np.ndarray, triangle:np.ndarray=None) -> None:
        """
        polygon : array of vertices of polygon
        """
        self.polygon = polygon
        self.dim = polygon.shape[-1]
        self.triangle = triangle
        self._setup_shapes()

    def _setup_shapes(self) -> None:
        """
        """
        if self.triangle is None:
            self.triangle = np.array([(-0.5,0), (0.5,0), (0,np.sqrt(3)/2)])
        self.polygon -= centroid(self.polygon)
        self.triangle -= centroid(self.triangle)
        self.unique_triples = unique_triples(self.polygon)

    def define_distribution(self, distribution=None, distribution_weights=None, alpha=[1,1,1], nsamples=10000) -> None:
        """

        Parameters
        ----------------
        distribution

        distribution_weights

        alpha: list, default: [1,1,1]

        nsamples: int, default: 10000
        """
        if distribution is None:
            dirichlet_points = np.random.dirichlet(alpha, nsamples)
            points = np.dot(dirichlet_points, self.triangle)
            self.distribution_weights = barycentric_weights(dirichlet_points)
            self.distribution = points[self.distribution_weights > 0]
        else:
            if distribution.shape[-1] != self.dim + 1:
                raise ValueError("No")
            self.distribution = distribution
            self.distribution_weights = distribution_weights

    def create_mapping(self, lerp_kind:int=0):
        """
        """
        self.kind = lerp_kind
        maps, matrices, rotated_polygons, triangulations = [], [], [], []
        for _ in range(12):
            for indices in self.unique_triples:
                for i in range(3):
                    dst_pts = self.map_polygon_to_triangle(np.roll(self.triangle.copy(), i, axis=0), indices)
                    if dst_pts is None:
                        continue
                    polygon_wc = list(self.polygon)
                    dst_wc = list(dst_pts)
                    polygon_wc.append(np.array([0,0]))
                    dst_wc.append(np.array([0,0]))
                    polygon_wc = np.asarray(polygon_wc)
                    dst_wc = np.asarray(dst_wc)
                    piecewise_matrices, triangulation = self._triangulation_method(polygon_wc, dst_wc)
                    matrices.append(piecewise_matrices)
                    triangulations.append(triangulation)
                    maps.append(dst_wc)
                    rotated_polygons.append(polygon_wc.copy())
            self.polygon = self.polygon @ rot_mat(np.pi/12)

        # print([len(m) for m in maps])
        self.maps = np.asarray(maps)
        # print([M.shape for M in matrices])
        self.matrices = np.asarray(matrices)
        self.triangulations = np.asarray(triangulations)
        self.rotated_polygons = np.asarray(rotated_polygons)
        return self.maps, self.matrices, self.triangulations, self.rotated_polygons

    def map_polygon_to_triangle(self, triangle:np.ndarray, tri_indices:np.ndarray):
        """
        """
        dst_pts = self.polygon.copy()
        dst_pts[tri_indices] = triangle
        if is_self_intersecting(dst_pts): 
            return None
        last_index = -1
        v1_idx, v2_idx = tri_indices[last_index%3], tri_indices[(last_index+1)%3]
        src_v1, src_v2 = self.polygon[v1_idx], self.polygon[v2_idx]
        dst_v1, dst_v2 = dst_pts[v1_idx], dst_pts[v2_idx]
        for j, p in enumerate(self.polygon):
            if j in tri_indices:
                last_index += 1
                v1_idx, v2_idx = tri_indices[last_index%3], tri_indices[(last_index+1)%3]
                src_v1, src_v2 = self.polygon[v1_idx], self.polygon[v2_idx]
                dst_v1, dst_v2 = dst_pts[v1_idx], dst_pts[v2_idx]
            else:
                dst_pts[j] = interpolate_vertex(p, src_v1, src_v2, dst_v1, dst_v2, kind=self.kind)
        return dst_pts

    def _triangulation_method(self, src_pts, dst_pts):
        tri_poly_dst = Delaunay(dst_pts)
        piecewise_matrices = create_piecewise_matrices(src_pts, dst_pts, tri_poly_dst.simplices)
        return piecewise_matrices, tri_poly_dst

    def dirichlet_energy(self, matrices:np.ndarray, mapping:np.ndarray, triangulation:Delaunay, polygon):
        areasA = unsigned_area(polygon[triangulation.simplices])
        areasB = unsigned_area(mapping[triangulation.simplices])
        spectral_norms = np.array([max(np.linalg.svd(J)[1])**2 * B + max(np.linalg.svd(np.linalg.inv(J))[1])**2 * A for J, A, B in zip(matrices, areasA, areasB)])
        return np.sum(spectral_norms)

    def minimize_jacobian(self):
        energies = np.array([self.dirichlet_energy(mats, maps, tris, polys) 
                             for mats, maps, tris, polys in zip(self.matrices, self.maps, self.triangulations, self.rotated_polygons)])
        optimal_index = np.argmin(energies)
        return optimal_index