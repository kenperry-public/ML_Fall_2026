import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

import ipywidgets as widgets
from IPython.display import display, clear_output

def majority_prob(Ms, p, include_tie=False):
    """
    P(X >= threshold) where X ~ Bin(M, p).
    include_tie=False  -> strict majority: X >= floor(M/2)+1
    include_tie=True   -> includes tie when M even: X >= ceil(M/2)
    """
    Ms = np.asarray(Ms, dtype=int)

    if include_tie:
        # threshold = ceil(M/2)
        thresh = (Ms + 1) // 2
    else:
        # threshold = floor(M/2)+1
        thresh = Ms // 2 + 1

    # P(X >= thresh) = sf(thresh - 1)
    return binom.sf(thresh - 1, Ms, p)

def plot(M_min, M_max, p, include_tie):
    Ms = np.arange(M_min, M_max + 1)
    probs = majority_prob(Ms, p, include_tie=include_tie)

    plt.figure(figsize=(9, 4.5))
    plt.plot(Ms, probs)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("M")
    plt.ylabel("P(majority correct)")
    title = "P(X ≥ ceil(M/2))" if include_tie else "P(X ≥ floor(M/2)+1)  (strict majority)"
    plt.title(f"{title},  p={p:.4f}")
    plt.grid(True, alpha=0.3)
    plt.show()

# Widgets
p_slider = widgets.FloatSlider(value=0.55, min=0.0, max=1.0, step=0.001, description="p", readout_format=".3f")
p_box    = widgets.BoundedFloatText(value=0.55, min=0.0, max=1.0, step=0.001, description="p input")
widgets.jslink((p_slider, "value"), (p_box, "value"))  # keep them synced

M_min = widgets.IntSlider(value=10, min=1, max=999, step=1, description="M min")
M_max = widgets.IntSlider(value=200, min=2, max=1000, step=1, description="M max")
include_tie = widgets.Checkbox(value=False, description="Include tie (M even)")

out = widgets.Output()

def refresh(_=None):
    with out:
        clear_output(wait=True)
        m1, m2 = M_min.value, M_max.value
        if m2 <= m1:
            print("Set M max > M min")
            return
        plot(m1, m2, p_slider.value, include_tie.value)

for w in [p_slider, p_box, M_min, M_max, include_tie]:
    w.observe(refresh, names="value")

display(widgets.VBox([
    widgets.HBox([p_slider, p_box]),
    widgets.HBox([M_min, M_max, include_tie]),
    out
]))

refresh()
