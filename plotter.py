import os
import torch.nn.functional as F

from io import BytesIO
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import map_scale

def plot_centers(img, epoch, id, centers):
    """Plots the sequence of centers the agent predicted for a single image"""
    plt.ioff() 

    out_dir = os.path.join('plots', f'epoch{epoch}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    x, y = [c[0] for c in centers], [c[1] for c in centers]

    img = img.numpy()
    if img.shape[1] == img.shape[2]:
        img = img.transpose(1,2,0)

    H, _, _ = img.shape

    # ----- Scale check ([-1, 1] --> [H, W]) -----
    if max(x) <= 1 and min(x) >= -1:
        x, y = [map_scale(x_val.item(), H) for x_val in x], [map_scale(y_val.item(), H) for y_val in y]

    # ----- Plotting -----
    fig, ax = plt.subplots()
    ax.axis('off')

    # plot image and points
    ax.imshow(img)

    #plot starting point and subsequent points
    ax.plot(x, y, color='red', marker='o', zorder=2) 
    ax.plot(x[0], y[0], color='white', marker='o', zorder=2)

    for i in range(len(centers) - 1):
        x_start, y_start = x[i], y[i]
        x_end, y_end = x[i+1], y[i+1]

        # plot arrows in the center of points
        x_diff =(x_end - x_start) / 2
        y_diff = (y_end - y_start) / 2

        ax.arrow(x_start, y_start, x_diff, y_diff, head_width=0.5, head_length=.75, fc='red', ec='red', zorder=1)

    ax.set_title("Sequence of Agents Vision")

    # ----- Save Plot -----
    save_path = os.path.join(out_dir, f'vision_plot_{id}.png')
    fig.savefig(save_path)

    plt.close()
    
def plot_attentions(distributions, epoch, id, action_space, cmap='inferno'):
    """Plots Probability Distribution of Next Action Over Action Space"""

    plt.ioff() 

    out_dir = os.path.join("sequences", f"epoch{epoch}", str(id))
    out_folder = os.path.join("sequences", f"epoch{epoch}")
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        os.mkdir(out_dir)
    elif not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # ----- Plot Probabilities of Action Space -----
    for idx, distribution in enumerate(distributions):

        # ----- Get Probs And Reshape -----
        dist = F.softmax(distribution, dim=-1)
        dist = dist.view(action_space, action_space).detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax.axis('off')

        # ----- Create Plot -----
        im   = ax.imshow(dist, cmap=cmap, interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', rotation=270, labelpad=10)
        
        # ----- Save Plot -----
        save_path = os.path.join(out_dir, f"attention_{idx+1}.png")
        fig.savefig(save_path)

        plt.close()

def make_gif(img, centers, patch_size, filename="attention.gif", fps=2):
    out_dir = os.path.join('gifs')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    x, y = [c[0] for c in centers], [c[1] for c in centers]

    img = img.numpy()
    if img.shape[1] == img.shape[2]:
        img = img.transpose(1,2,0)

    H, _, _ = img.shape

    # ----- Scale check ([-1, 1] --> [H, W]) -----
    if max(x) <= 1 and min(x) >= -1:
        x, y = [map_scale(x_val.item(), H) for x_val in x], [map_scale(y_val.item(), H) for y_val in y]

    # ----- Create and add frames -----
    frames = []
    for i in range(len(x)):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(img)

        x_corner = x[i] - (patch_size / 2)
        y_corner = y[i] - (patch_size / 2)
        box = patches.Rectangle((x_corner, y_corner), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(box)
        ax.plot(x[i], y[i], 'o', color='red', markersize=5, label='Center')


        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    # ----- create and save gif -----
    gif_path = os.path.join(out_dir, filename)
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {gif_path}")