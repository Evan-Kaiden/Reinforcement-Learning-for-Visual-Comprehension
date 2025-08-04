import matplotlib.pyplot as plt
from utils import map_scale
def plot_centers(img, centers):
    """Plots the sequence of centers the agent predicted for a single image"""
    x, y = [c[0] for c in centers], [c[1] for c in centers]

    if img.shape[1] == img.shape[2]:
        img = img.transpose(1,2,0)

    H, _, _ = img.shape

    if max(x) <= 1 and min(x) >= -1: # scale is [-1, 1]
        x, y = [map_scale(x_val.item(), H) for x_val in x], [map_scale(y_val.item(), H) for y_val in y]

    fig, ax = plt.subplots()

    # plot image and points
    ax.imshow(img)
    ax.plot(x, y, color='red', marker='o', zorder=2) 

    for i in range(len(centers) - 1):
        x_start, y_start = x[i], y[i]
        x_end, y_end = x[i+1], y[i+1]

        # plot arrows in the center of two points
        x_diff =(x_end - x_start) / 2
        y_diff = (y_end - y_start) / 2

        ax.arrow(x_start, y_start, x_diff, y_diff, head_width=0.5, head_length=.75, fc='red', ec='red', zorder=1)

    ax.set_title("Sequence of Agents Glimpses")
    plt.show()

def plot_patches(img, patches):
    pass