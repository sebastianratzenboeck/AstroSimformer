import torch


def fitzpatrick99_k_lambda(wav_aa, Rv=3.1):
    """
    Fitzpatrick (1999) extinction curve in torch.
    Args:
        wav_aa : (M,) Wavelengths in Angstrom
        Rv     : scalar float, default 3.1
    Returns:
        k_lambda = A_lambda / E(B-V), shape (M,)
    """
    x = 1e4 / wav_aa  # inverse micron
    k = torch.zeros_like(x)

    device = x.device

    uv_mask = (x >= 1.e4 / 2700.)
    x0, gamma = 4.596, 0.99
    c3, c4, c5 = 3.23, 0.41, 5.9
    c2 = -0.824 + 4.717 / Rv
    c1 = 2.030 - 3.007 * c2

    y = x[uv_mask]
    d = y**2 / ((y**2 - x0**2)**2 + y**2 * gamma**2)
    f = torch.zeros_like(y)
    f[y >= c5] = 0.5392 * (y[y >= c5] - c5)**2 + 0.05644 * (y[y >= c5] - c5)**3
    k[uv_mask] = c1 + c2 * y + c3 * d + c4 * f

    x_uv_spline = 1e4 / torch.tensor([2700., 2600.], device=device)
    d_uv = x_uv_spline**2 / ((x_uv_spline**2 - x0**2)**2 + x_uv_spline**2 * gamma**2)
    k_uv_spline = c1 + c2 * x_uv_spline + c3 * d_uv

    anchors_x = 1e4 / torch.tensor([float('inf'), 26500., 12200., 6000., 5470., 4670., 4110.], device=device)
    anchors_k = torch.tensor([
        0.,
        0.26469 * Rv / 3.1,
        0.82925 * Rv / 3.1,
        -0.422809 + 1.00270 * Rv + 2.13572e-04 * Rv**2,
        -5.13540e-02 + 1.00216 * Rv - 7.35778e-05 * Rv**2,
        0.700127 + 1.00184 * Rv - 3.32598e-05 * Rv**2,
        1.19456 + 1.01707 * Rv - 5.46959e-03 * Rv**2 + 7.97809e-04 * Rv**3 - 4.45636e-05 * Rv**4
    ], device=device) - Rv

    anchors_x = torch.cat([anchors_x, x_uv_spline])
    anchors_k = torch.cat([anchors_k, k_uv_spline])

    y = x[~uv_mask]
    k[~uv_mask] = torch_interpolate(y, anchors_x, anchors_k)
    k_lam = k + Rv
    return k_lam / 3.1


def torch_interpolate(x, x_points, y_points, eps=1e-7):
    right_idx = torch.searchsorted(x_points, x)
    left_idx = (right_idx - 1).clamp(min=0)
    right_idx = right_idx.clamp(max=x_points.shape[0] - 1)

    left_x = x_points[left_idx]
    right_x = x_points[right_idx]
    left_y = y_points[left_idx]
    right_y = y_points[right_idx]

    left_dist = x - left_x
    total_dist = (right_x - left_x).clamp(min=eps)

    interp = left_y + (left_dist / total_dist) * (right_y - left_y)
    return interp