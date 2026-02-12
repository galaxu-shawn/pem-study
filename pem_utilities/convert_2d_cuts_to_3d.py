# import numpy as np

# from .utils import FullPattern, HorizontalSlice, VerticalSlice, build_equiangular_grid


# def summing(
#     vertical_slice: VerticalSlice,
#     horizonal_slice: HorizontalSlice,
# ) -> FullPattern:
#     r"""Interpolate a full 3-D pattern from a vertical and horizontal slice using the summing algorithm.

#     Parameters
#     ----------
#     vertical_slice : VerticalSlice
#     horizontal_slice : HorizontalSlice

#     Returns
#     -------
#     FullPattern
#     """
#     theta_resolution = (vertical_slice.size // 2) + 1
#     return 0.5 * (
#         vertical_slice[:theta_resolution, np.newaxis] + horizonal_slice[np.newaxis, :]
#     )


# def bilinear(
#     vertical_slice: VerticalSlice,
#     horizonal_slice: HorizontalSlice,
# ) -> FullPattern:
#     r"""Interpolate a full 3-D pattern from a vertical and horizontal slice using the bilinear algorithm.

#     Parameters
#     ----------
#     vertical_slice : VerticalSlice
#     horizontal_slice : HorizontalSlice

#     Returns
#     -------
#     FullPattern
#     """
#     phi_resolution = horizonal_slice.size
#     theta_resolution = (vertical_slice.size // 2) + 1

#     fnt_plane = vertical_slice[:theta_resolution, np.newaxis]
#     bck_plane = np.flip(
#         np.concatenate((vertical_slice[theta_resolution - 1 :], vertical_slice[0:1]))
#     )[:, np.newaxis]

#     phi = np.linspace(0, 2 * np.pi, phi_resolution, endpoint=False)
#     fnt_plane_dist = phi
#     fnt_plane_dist[phi >= np.pi] = 2 * np.pi - fnt_plane_dist[phi >= np.pi]
#     fnt_plane_dist = fnt_plane_dist[np.newaxis, :]
#     bck_plane_dist = np.pi - fnt_plane_dist

#     equator = horizonal_slice[np.newaxis, :]

#     theta = np.linspace(0, np.pi, theta_resolution)
#     equator_dist = theta
#     equator_dist[theta >= np.pi / 2] = np.pi - theta[theta >= np.pi / 2]
#     equator_dist = equator_dist[:, np.newaxis]
#     pole_dist = np.pi / 2 - equator_dist

#     north_pole = vertical_slice[0]  # pyright: ignore[reportAny]
#     south_pole = vertical_slice[theta_resolution - 1]  # pyright: ignore[reportAny]
#     pole = np.ones((theta_resolution, phi_resolution))
#     pole[theta < np.pi / 2, :] *= north_pole
#     pole[theta >= np.pi / 2, :] *= south_pole

#     full_pattern = (
#         fnt_plane_dist * bck_plane
#         + bck_plane_dist * fnt_plane
#         + pole_dist * equator
#         + equator_dist * pole
#     ) / (fnt_plane_dist + bck_plane_dist + equator_dist + pole_dist)

#     return full_pattern


# def weighted_bilinear(
#     vertical_slice: VerticalSlice,
#     horizonal_slice: HorizontalSlice,
# ) -> FullPattern:
#     r"""Interpolate a full 3-D pattern from a vertical and horizontal slice using the weighted bilinear algorithm.

#     Parameters
#     ----------
#     vertical_slice : VerticalSlice
#     horizontal_slice : HorizontalSlice

#     Returns
#     -------
#     FullPattern
#     """
#     phi_resolution = horizonal_slice.size
#     theta_resolution = (vertical_slice.size // 2) + 1
#     phi_grid, theta_grid = build_equiangular_grid(phi_resolution, theta_resolution)

#     fnt_plane = vertical_slice[:theta_resolution, np.newaxis]
#     bck_plane = np.flip(
#         np.concatenate((vertical_slice[theta_resolution - 1 :], vertical_slice[0:1]))
#     )[:, np.newaxis]

#     phi = np.linspace(0, 2 * np.pi, phi_resolution, endpoint=False)
#     fnt_plane_dist = phi
#     fnt_plane_dist[phi >= np.pi] = 2 * np.pi - fnt_plane_dist[phi >= np.pi]
#     fnt_plane_dist = fnt_plane_dist[np.newaxis, :]
#     bck_plane_dist = np.pi - fnt_plane_dist

#     equator = horizonal_slice[np.newaxis, :]

#     theta = np.linspace(0, np.pi, theta_resolution)
#     equator_dist = theta
#     equator_dist[theta >= np.pi / 2] = np.pi - theta[theta >= np.pi / 2]
#     equator_dist = equator_dist[:, np.newaxis]
#     pole_dist = np.pi / 2 - equator_dist

#     north_pole = vertical_slice[0]  # pyright: ignore[reportAny]
#     south_pole = vertical_slice[theta_resolution - 1]  # pyright: ignore[reportAny]
#     pole = np.ones((theta_resolution, phi_resolution))
#     pole[theta < np.pi / 2, :] *= north_pole
#     pole[theta >= np.pi / 2, :] *= south_pole

#     weight_theta = (pole_dist * equator_dist) / np.square(pole_dist + equator_dist)
#     weight_phi = (fnt_plane_dist * bck_plane_dist) / np.square(
#         fnt_plane_dist + bck_plane_dist
#     )

#     with np.errstate(divide="ignore", invalid="ignore"):
#         full_pattern = (
#             (fnt_plane_dist * bck_plane + bck_plane_dist * fnt_plane) * weight_theta
#             + (pole_dist * equator + equator_dist * pole) * weight_phi
#         ) / (
#             (fnt_plane_dist + bck_plane_dist) * weight_theta
#             + (equator_dist + pole_dist) * weight_phi
#         )

#     nan_mask = np.isnan(full_pattern)
#     full_pattern[nan_mask & (theta_grid == 0)] = north_pole
#     full_pattern[nan_mask & (theta_grid == np.pi)] = south_pole
#     full_pattern[nan_mask & (phi_grid == 0)] = horizonal_slice[0]
#     full_pattern[nan_mask & (phi_grid == np.pi)] = horizonal_slice[phi_resolution // 2]

#     return full_pattern


# def horizontal_projection(
#     vertical_slice: VerticalSlice,
#     horizonal_slice: HorizontalSlice,
# ) -> FullPattern:
#     r"""Interpolate a full 3-D pattern from a vertical and horizontal slice using the horizontal projection algorithm.

#     Altair's equation for HPI uses (azimuth, elevation) rather than (phi, theta) -> Gh(0) = Gv(0); Gh(pi) = Gv(pi).

#     Parameters
#     ----------
#     vertical_slice : VerticalSlice
#     horizontal_slice : HorizontalSlice

#     Returns
#     -------
#     FullPattern
#     """
#     phi_resolution = horizonal_slice.size
#     theta_resolution = (vertical_slice.size // 2) + 1

#     fnt_plane = vertical_slice[:theta_resolution, np.newaxis]
#     bck_plane = np.flip(
#         np.concatenate((vertical_slice[theta_resolution - 1 :], vertical_slice[0:1]))
#     )[:, np.newaxis]

#     equator = horizonal_slice[np.newaxis, :]

#     north_pole = vertical_slice[0]  # pyright: ignore[reportAny]
#     south_pole = vertical_slice[theta_resolution - 1]  # pyright: ignore[reportAny]

#     phi = np.linspace(0, 2 * np.pi, phi_resolution, endpoint=False)
#     norm_fnt_plane_dist = phi
#     norm_fnt_plane_dist[phi >= np.pi] = 2 * np.pi - norm_fnt_plane_dist[phi >= np.pi]
#     norm_fnt_plane_dist = (1 / np.pi) * norm_fnt_plane_dist[np.newaxis, :]
#     norm_bck_plane_dist = 1 - norm_fnt_plane_dist

#     full_pattern = equator - (  # pyright: ignore[reportAny]
#         norm_bck_plane_dist * (north_pole - fnt_plane)
#         + norm_fnt_plane_dist * (south_pole - bck_plane)
#     )

#     return full_pattern  # pyright: ignore[reportAny]


# def exponential(
#     vertical_slice: VerticalSlice,
#     horizonal_slice: HorizontalSlice,
# ) -> FullPattern:
#     r"""Interpolate a full 3-D pattern from a vertical and horizontal slice using the exponential algorithm.

#     Parameters
#     ----------
#     vertical_slice : VerticalSlice
#     horizontal_slice : HorizontalSlice

#     Returns
#     -------
#     FullPattern
#     """
#     theta_resolution = (vertical_slice.size // 2) + 1

#     equator = horizonal_slice[np.newaxis, :]
#     fnt_plane = vertical_slice[:theta_resolution, np.newaxis]
#     bck_plane = np.flip(
#         np.concatenate((vertical_slice[theta_resolution - 1 :], vertical_slice[0:1]))
#     )[:, np.newaxis]

#     max_gain = np.max(vertical_slice)

#     pole_diff = (
#         vertical_slice[3 * (theta_resolution - 1) // 2]
#         - vertical_slice[(theta_resolution - 1) // 2]
#     )

#     fnt_bck_ratio = bck_plane - fnt_plane

#     if pole_diff == 0:
#         fnt_bck_ratio[fnt_bck_ratio == 0] = 1
#         fnt_bck_ratio[fnt_bck_ratio != 0] = fnt_bck_ratio[fnt_bck_ratio != 0] * 1e-6
#     else:
#         fnt_bck_ratio /= pole_diff

#     # in case ratio is close to zero
#     full_pattern = (
#         fnt_plane + (1 / fnt_bck_ratio) * (bck_plane - fnt_plane) * equator - max_gain
#     )

#     return full_pattern


if __name__ == "__main__":
    import os
    import sys

    # Add the parent directory to the system path to import FarFields
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import far_fields as ff_module
    test_pattern = 'C:\\Users\\asligar\\OneDrive - ANSYS, Inc\\Documents\\Scripting\\github\\perceive_em\\antenna_device_library\\AWR_1642\\rx1.ffd'

    ff = ff_module.FarFields()

    ff.read_ffd(test_pattern,create_farfield_mesh=True)

    ff.plot()