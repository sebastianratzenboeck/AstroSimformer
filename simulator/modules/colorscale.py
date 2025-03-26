import plotly.express as px
import numpy as np


class ColorScale:
    def __init__(self, min_val, max_val, dx=None, cmap='rdbu', n_colors=10, mpl=True):
        self.min_value = min_val
        self.max_value = max_val
        self.cmap = cmap
        self.unique_colors = self.make_unique_colors()
        self.n_colors = np.min((n_colors, len(self.unique_colors)))
        self.uq = np.linspace(0, 1, self.n_colors)
        self.mpl = mpl
        if dx is None:
            self.dx = (self.max_value - self.min_value) / (self.n_colors - 1)
        else:
            self.dx = dx
        # Set mappings
        self.color_map = []
        self.color_scale = []
        self.make_colorscale()

    def make_unique_colors(self):
        unique_colors = [cinfo[1] for cinfo in px.colors.get_colorscale(self.cmap)][::-1]
        color_max = [unique_colors[-1]]
        return unique_colors + color_max

    def make_colorscale(self):
        self.color_map = []
        self.color_scale = []
        curr_age = self.min_value
        for u_color, start, stop in zip(self.unique_colors, self.uq[:-1], self.uq[1:]):
            # construct colorscale
            self.color_scale.append([start, u_color])
            self.color_scale.append([stop, u_color])
            # construct color_ages
            self.color_map.append({'range': [curr_age, curr_age + self.dx], 'color': u_color})
            # increment age
            curr_age += self.dx

        self.color_map[-1]['range'][1] = self.max_value
        self.color_map[0]['range'][0] = self.min_value
        return

    @staticmethod
    def isin_constrained_range(value, range_vals):
        lo = np.min(range_vals)
        hi = np.max(range_vals)
        return (value >= lo) & (value < hi)

    @staticmethod
    def rgb2frac(rgb_string):
        integers = rgb_string[rgb_string.find("(") + 1:rgb_string.find(")")].split(",")
        # Converting string integers to actual integers
        integers = [float(num) / 255 for num in integers]
        return integers

    def get_color(self, value):
        for color_info in self.color_map:
            if self.isin_constrained_range(value=value, range_vals=color_info['range']):
                if self.mpl:
                    return self.rgb2frac(color_info['color'])
                else:
                    return color_info['color']
        return None
