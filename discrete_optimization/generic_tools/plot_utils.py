#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Contains common utilities to plot solution,
for now it's mainly to patch API break happening with matplotlib3.9.0 on colors.
"""
import matplotlib
import matplotlib.cm


def get_cmap_with_nb_colors(color_map_str: str, nb_colors: int = 2):
    try:
        return matplotlib.colormaps[color_map_str].resampled(nb_colors)
    except:
        return matplotlib.cm.get_cmap(color_map_str, nb_colors)


def get_cmap(color_map_str: str):
    try:
        return matplotlib.colormaps[color_map_str]
    except:
        return matplotlib.cm.get_cmap(color_map_str)
