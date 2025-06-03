import torch
from numpy.lib.recfunctions import structured_to_unstructured
import numpy as np

default_rng = np.random.default_rng(135)


def position(cloud):
    """Cloud to point positions (xyz)."""
    if cloud.dtype.names:
        x = structured_to_unstructured(cloud[['x', 'y', 'z']])
    else:
        x = cloud
    return x


def within_bounds(x, min=None, max=None, bounds=None, log_variable=None):
    """Mask of x being within bounds  min <= x <= max."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    assert isinstance(x, torch.Tensor)

    keep = torch.ones((x.numel(),), dtype=torch.bool, device=x.device)

    if bounds:
        assert min is None and max is None
        min, max = bounds

    if min is not None and min > -float('inf'):
        if not isinstance(min, torch.Tensor):
            min = torch.tensor(min)
        keep = keep & (x.flatten() >= min)
    if max is not None and max < float('inf'):
        if not isinstance(max, torch.Tensor):
            max = torch.tensor(max)
        keep = keep & (x.flatten() <= max)

    if log_variable is not None:
        print('%.3f = %i / %i points kept (%.3g <= %s <= %.3g).'
              % (keep.double().mean(), keep.sum(), keep.numel(),
                 min if min is not None else float('nan'),
                 log_variable,
                 max if max is not None else float('nan')))

    return keep


def filter_grid(cloud, grid_res, keep='first', log=False, rng=default_rng, only_mask=False):
    """
    Keep single point within each cell. Order is not preserved.
    """
    assert isinstance(cloud, np.ndarray), type(cloud)
    # assert cloud.dtype.names
    assert isinstance(grid_res, (float, int)) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    if cloud.dtype.names:
        cloud = cloud.ravel()
    if keep == 'first':
        pass
    elif keep == 'random':
        rng.shuffle(cloud)
    elif keep == 'last':
        cloud = cloud[::-1]

    x = position(cloud)
    keys = np.floor(x / grid_res).astype(int)
    assert keys.size > 0
    _, ind = np.unique(keys, return_index=True, axis=0)

    if log:
        print('%.3f = %i / %i points kept (grid res. %.3f m).'
              % (len(ind) / len(keys), len(ind), len(keys), grid_res))

    if only_mask:
        return ind

    filtered = cloud[ind]
    return filtered

def estimate_heightmap(points, grid_res, d_max, h_max, r_min=None, h_min=None):
    # remove nans from the point cloud if any
    mask = ~torch.isnan(points).any(dim=1)
    points = points[mask]

    if r_min is not None:
        # remove points in a r_min radius
        distances = torch.norm(points[:, :2], dim=1)
        mask = distances > r_min
        points = points[mask]

    if h_min is None:
        h_min = -h_max

    mask = ((points[:, 0] > -d_max) & (points[:, 0] < d_max) &
            (points[:, 1] > -d_max) & (points[:, 1] < d_max) &
            (points[:, 2] > h_min) & (points[:, 2] < h_max))
    points = points[mask]

    # Extract X, Y, Z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute grid dimensions
    x_bins = torch.arange(-d_max, d_max, grid_res)
    y_bins = torch.arange(-d_max, d_max, grid_res)

    # Digitize coordinates to find grid indices
    x_indices = torch.bucketize(x.contiguous(), x_bins) - 1
    y_indices = torch.bucketize(y.contiguous(), y_bins) - 1

    # Use scatter_reduce to populate the heightmap
    flat_indices = y_indices * len(x_bins) + x_indices  # Flattened indices
    flat_heightmap = torch.full((len(y_bins) * len(x_bins),), float('nan'))

    # Use scatter_reduce to take the maximum height per grid cell
    flat_heightmap = torch.scatter_reduce(
        flat_heightmap,
        dim=0,
        index=flat_indices,
        src=z,
        reduce="amax",
        include_self=False
    )

    # Reshape back to 2D
    heightmap = flat_heightmap.view(len(y_bins), len(x_bins))

    # Replace NaNs with a default value (e.g., 0.0)
    measurements_mask = ~torch.isnan(heightmap)
    heightmap = torch.nan_to_num(heightmap, nan=0.0)
    # heightmap = torch.nan_to_num(heightmap, nan=(h_max + h_min) / 2.)

    hm = torch.stack([heightmap, measurements_mask], dim=0)  # (2, H, W)

    return hm


def hm_to_cloud(height, d_max, mask=None):
    assert isinstance(height, np.ndarray) or isinstance(height, torch.Tensor)
    assert height.ndim == 2
    if mask is not None:
        assert isinstance(mask, (np.ndarray, torch.Tensor))
        assert mask.ndim == 2
        assert height.shape == mask.shape
        mask = mask.bool() if isinstance(mask, torch.Tensor) else mask.astype(bool)
    z_grid = height
    if isinstance(height, np.ndarray):
        x_grid = np.linspace(-d_max, d_max, z_grid.shape[0])
        y_grid = np.linspace(-d_max, d_max, z_grid.shape[1])
        x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing='xy')
        hm_cloud = np.stack([x_grid, y_grid, z_grid], axis=2)
    else:
        x_grid = torch.linspace(-d_max, d_max, z_grid.shape[0]).to(z_grid.device)
        y_grid = torch.linspace(-d_max, d_max, z_grid.shape[1]).to(z_grid.device)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='xy')
        hm_cloud = torch.stack([x_grid, y_grid, z_grid], dim=2)
    if mask is not None:
        hm_cloud = hm_cloud[mask]
    hm_cloud = hm_cloud.reshape([-1, 3])
    return hm_cloud


def filter_column(cloud : np.ndarray,
                  d_max : float,
                  pose=None,
                  size=None,
                  phi=None,
                  prob=0.5) -> np.ndarray:
    """ Mask out a column of points in a 3D point cloud.
    :param points: (N, 3) array of points in the point cloud.
    :param d_max: float, maximum distance from the origin to the column center.
    :param prob: float, probability of removing a column.
    :return: boolean mask of shape (N,) where True indicates that the point is outside the column.
    """
    if np.random.rand() > prob:
        # Do not remove a column
        mask = np.ones(len(cloud), dtype=bool)
        return mask

    # Ensure points are in (N, 3) format (x, y, z)
    points = position(cloud)

    if pose is None:
        pose = np.random.uniform(-d_max*0.8, d_max*0.8, (2,))
    if size is None:
        size = np.random.uniform(0.4, 0.6, (2,)) * d_max
    if phi is None:
        phi = np.random.uniform(-np.pi, np.pi)

    # A column pose (x, y, phi) and size (dx, dy)
    x, y = pose
    dx, dy = size
    # print(f'Column center: ({x:.2f}, {y:.2f}), dx: {dx:.2f}, dy: {dy:.2f}, phi: {phi:.2f}')
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])

    # Rotate the points
    points_rot = points @ Rz.T

    # Rotate the mask center (x, y) into the rotated frame
    xy_rot = np.array([x, y, 0]) @ Rz.T

    # Apply the mask in the rotated frame
    column_mask = (np.abs(points_rot[:, 0] - xy_rot[0]) > dx / 2) | \
                  (np.abs(points_rot[:, 1] - xy_rot[1]) > dy / 2)
    return column_mask