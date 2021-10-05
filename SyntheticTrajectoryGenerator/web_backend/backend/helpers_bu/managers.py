class SegementManager:
    def __init__(self):
        pass

    def spatial_segmentation(
        self, segments, min_dist: float, measure: str = "euclidean"
    ):
        final_segments = []
        for segment in segments:
            final_segments.append([])
            for point in segment:
                if measure == "euclidean":
                    if point.dx_euclidean > min_dist:
                        final_segments.append([])
                elif measure == "manhatten":
                    if point.dx_manhatten > min_dist:
                        final_segments.append([])
                final_segments[-1].append(point)
        return final_segments

    def temporal_segmentation(self, segments, min_time: float):
        final_segments = []
        for segment in segments:
            final_segments.append([])
            for point in segment:
                if point.dt > min_time:
                    final_segments.append([])
                final_segments[-1].append(point)
        return final_segments
