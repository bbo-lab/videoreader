from svidreader.video_supplier import VideoSupplier
import numpy as np
from scipy.spatial import Delaunay

class ColorMapper(VideoSupplier):
    def __init__(self, reader, source_colors, destination_colors):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.cache = (None,None)
        # Ensure inputs are numpy arrays
        source_colors = np.array(source_colors)
        destination_colors = np.array(destination_colors)

        # Create a Delaunay triangulation for the source colors
        delaunay = Delaunay(source_colors)

        def map_colors(query_colors):
            query_colors = np.atleast_2d(query_colors)  # Ensure input is 2D
            # Find simplices (triangles) for each query color
            simplices = delaunay.find_simplex(query_colors)

            # Handle colors outside the convex hull
            outside_mask = simplices == -1
            if np.any(outside_mask):
                query_colors_outside = query_colors[outside_mask]

                # Compute distances to all source colors
                distances = np.linalg.norm(
                    source_colors[None, :, :] - query_colors_outside[:, None, :],
                    axis=2
                )

                # Find the two nearest neighbors
                nearest_indices = np.argsort(distances, axis=1)[:, :2]
                nearest_source = source_colors[nearest_indices]
                nearest_dest = destination_colors[nearest_indices]

                # Compute weights for interpolation
                weights = 1 / distances[:, nearest_indices]
                weights = weights / weights.sum(axis=1, keepdims=True)

                # Interpolate using the weights
                outside_colors = (weights[:, :, None] * nearest_dest).sum(axis=1)
            else:
                outside_colors = np.empty((0, destination_colors.shape[1]))

            # Handle colors inside the convex hull
            inside_mask = ~outside_mask
            inside_colors = np.empty((0, destination_colors.shape[1]))
            if np.any(inside_mask):
                query_colors_inside = query_colors[inside_mask]
                simplices_inside = simplices[inside_mask]

                # Extract vertices and transform matrices for simplices
                vertices = delaunay.simplices[simplices_inside]
                transform = delaunay.transform[simplices_inside]

                # Compute barycentric coordinates
                deltas = query_colors_inside - transform[:, 3]
                bary_coords = np.einsum('ijk,ik->ij', transform[:, :3], deltas)

                # Include the weight for the last vertex (1 - sum(barycentric))
                bary_coords = np.hstack([bary_coords, 1 - bary_coords.sum(axis=1, keepdims=True)])

                # Interpolate destination colors
                destination_vertices = destination_colors[vertices]
                inside_colors = np.einsum('ij,ijk->ik', bary_coords, destination_vertices)

            # Combine inside and outside results
            mapped_colors = np.zeros((query_colors.shape[0], destination_colors.shape[1]))
            mapped_colors[outside_mask] = outside_colors
            mapped_colors[inside_mask] = inside_colors

            return mapped_colors
        self.map_colors = map_colors


    def read(self, index, force_type=np):
        current = self.inputs[0].read(index=index, force_type=np)
        current = self.map_colors(current.reshape(-1,3)).reshape(current.shape)
        return current
