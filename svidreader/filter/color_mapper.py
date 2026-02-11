from svidreader.video_supplier import VideoSupplier
import numpy as np
from scipy.spatial import Delaunay
import logging

logger = logging.getLogger(__name__)

class ColorMapperFunctional:
    def __init__(self, names, source_colors, destination_colors):
        self.names = names
        self.source_colors = np.array(source_colors, dtype=float)
        self.destination_colors = np.array(destination_colors, dtype=float)
        self.delaunay = Delaunay(self.source_colors)

    def __call__(self, query_colors):
        query_colors = np.asarray(query_colors, dtype=float)
        shape = query_colors.shape[:-1]
        num_entries = np.prod(shape)
        num_input_channels = self.source_colors.shape[-1]
        num_output_channels = self.destination_colors.shape[1]
        assert query_colors.shape[-1] == num_input_channels, f"Expected query colors to have shape (*, {num_input_channels}), but got {query_colors.shape}"

        query_colors = query_colors.reshape(-1, num_input_channels)
        #print("minmax", query_colors.min(axis=0), query_colors.max(axis=0))
        mapped_colors = np.empty((num_entries, num_output_channels), dtype=float)
        # Find simplices (triangles) for each query color
        simplices = self.delaunay.find_simplex(query_colors)

        # Handle colors outside the convex hull
        outside_mask = simplices == -1
        inside_mask = ~outside_mask

        #print how much is inside and outside the convex hull
        logger.log(logging.INFO, f"Mapping colors: {np.sum(inside_mask)} inside, {np.sum(outside_mask)} outside the convex hull")

        if np.any(outside_mask):
            qc_out = query_colors[outside_mask]

            # Precompute simplex centroids ONCE if you want to optimize
            simplex_centers = self.source_colors[self.delaunay.simplices].mean(axis=1)

            dists = np.linalg.norm(
                simplex_centers[None, :, :] - qc_out[:, None, :],
                axis=2
            )

            nearest_simplex = np.argmin(dists, axis=1)

            vertices = self.delaunay.simplices[nearest_simplex]
            transform = self.delaunay.transform[nearest_simplex]

            deltas = qc_out - transform[:, -1]
            bary = np.einsum('ijk,ik->ij', transform[:, :-1], deltas)
            bary = np.hstack([bary, 1 - bary.sum(axis=1, keepdims=True)])

            dest_vertices = self.destination_colors[vertices]
            outside_colors = np.einsum(
                'ij,ijk->ik', bary, dest_vertices
            )

            mapped_colors[outside_mask] = outside_colors

        # Handle colors inside the convex hull
        if np.any(inside_mask):
            query_colors_inside = query_colors[inside_mask]
            simplices_inside = simplices[inside_mask]

            # Extract vertices and transform matrices for simplices
            vertices = self.delaunay.simplices[simplices_inside]
            transform = self.delaunay.transform[simplices_inside]

            # Compute barycentric coordinates
            deltas = query_colors_inside - transform[:, -1]
            bary_coords = np.einsum('ijk,ik->ij', transform[:, :-1], deltas)

            # Include the weight for the last vertex (1 - sum(barycentric))
            bary_coords = np.hstack([bary_coords, 1 - bary_coords.sum(axis=1, keepdims=True)])

            # Interpolate destination colors
            destination_vertices = self.destination_colors[vertices]
            inside_colors = np.einsum('ij,ijk->ik', bary_coords, destination_vertices)

            mapped_colors[inside_mask] = inside_colors

        return mapped_colors.reshape(*shape, num_output_channels)

    def as_pandas(self):
        import pandas as pd
        df = pd.DataFrame({
            "structure": list(self.names),
            "source_color": list(self.source_colors),
            "destination_color": list(self.destination_colors)
        })
        return df


class ColorMapper(VideoSupplier):
    def __init__(self, reader, source_colors, destination_colors):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.map_colors = self.get_color_mapper(source_colors, destination_colors)

    @staticmethod
    def get_color_mapper(
            source_colors:np.ndarray|list,
            destination_colors:np.ndarray|list,
            names=None):
        return ColorMapperFunctional(names=names, source_colors=source_colors, destination_colors=destination_colors)

    def read(self, index, force_type=np):
        current = self.inputs[0].read(index=index, force_type=np)
        current = self.map_colors(current)
        return current
