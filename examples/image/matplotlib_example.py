"""
Example using animations
===============================
"""
# %%
# Example of making sliders in Matplotlib (source: https://sphinx-gallery.github.io/stable/auto_examples/plot_8_animations.html)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Adapted from
# https://matplotlib.org/gallery/animation/basic_example.html


def _update_line(num):
    line.set_data(data[..., :num])
    return (line,)


fig, ax = plt.subplots()
data = np.random.RandomState(0).rand(2, 25)
(line,) = ax.plot([], [], "r-")
ax.set(xlim=(0, 1), ylim=(0, 1))
ani = animation.FuncAnimation(fig, _update_line, 25, interval=100, blit=True)
