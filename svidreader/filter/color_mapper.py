from svidreader.video_supplier import VideoSupplier
import numpy as np
from scipy.spatial import Delaunay

class ColorMapper(VideoSupplier):
    def __init__(self, reader, source_colors, destination_colors):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.map_colors = self.get_color_mapper(source_colors, destination_colors)

    @staticmethod
    def get_color_mapper(source_colors:np.ndarray|list, destination_colors:np.ndarray|list):
        # Create a Delaunay triangulation for the source colors

        source_colors = np.array(source_colors)
        destination_colors = np.array(destination_colors)
        delaunay = Delaunay(source_colors)
        def map_colors(query_colors):
            query_colors = np.atleast_2d(query_colors)  # Ensure input is 2D

            mapped_colors = np.empty((*query_colors.shape[:-1], destination_colors.shape[1]))
            # Find simplices (triangles) for each query color
            simplices = delaunay.find_simplex(query_colors)

            # Handle colors outside the convex hull
            outside_mask = simplices == -1
            inside_mask = ~outside_mask

            if np.any(outside_mask):
                qc_out = query_colors[outside_mask]

                # Precompute simplex centroids ONCE if you want to optimize
                simplex_centers = source_colors[delaunay.simplices].mean(axis=1)

                dists = np.linalg.norm(
                    simplex_centers[None, :, :] - qc_out[:, None, :],
                    axis=2
                )

                nearest_simplex = np.argmin(dists, axis=1)

                vertices = delaunay.simplices[nearest_simplex]
                transform = delaunay.transform[nearest_simplex]

                deltas = qc_out - transform[:, -1]
                bary = np.einsum('ijk,ik->ij', transform[:, :-1], deltas)
                bary = np.hstack([bary, 1 - bary.sum(axis=1, keepdims=True)])

                dest_vertices = destination_colors[vertices]
                outside_colors = np.einsum(
                    'ij,ijk->ik', bary, dest_vertices
                )

                mapped_colors[outside_mask] = outside_colors

            # Handle colors inside the convex hull
            inside_colors = np.empty((0, destination_colors.shape[1]))
            if np.any(inside_mask):
                query_colors_inside = query_colors[inside_mask]
                simplices_inside = simplices[inside_mask]

                # Extract vertices and transform matrices for simplices
                vertices = delaunay.simplices[simplices_inside]
                transform = delaunay.transform[simplices_inside]

                # Compute barycentric coordinates
                deltas = query_colors_inside - transform[:, -1]
                bary_coords = np.einsum('ijk,ik->ij', transform[:, :-1], deltas)

                # Include the weight for the last vertex (1 - sum(barycentric))
                bary_coords = np.hstack([bary_coords, 1 - bary_coords.sum(axis=1, keepdims=True)])

                # Interpolate destination colors
                destination_vertices = destination_colors[vertices]
                inside_colors = np.einsum('ij,ijk->ik', bary_coords, destination_vertices)

                mapped_colors[inside_mask] = inside_colors

            return mapped_colors
        return map_colors

    def read(self, index, force_type=np):
        current = self.inputs[0].read(index=index, force_type=np)
        current = self.map_colors(current.reshape(-1,3)).reshape(current.shape)
        return current
