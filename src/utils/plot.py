from typing import Optional, Literal, List
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from utils.tensor import to_numpy

DEFAULT_FIGSIZE = (16, 10)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def plot(
    data: torch.Tensor,
    title: Optional[str]=None
) -> Figure:
    """
    Plots a 1D sequence using matplotlib.

    Args:
        data (torch.Tensor): A 1D tensor representing the data to be plotted.
        title (Optional[str]): Optional title for the plot.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the plot.
    """
    if isinstance(data, torch.Tensor):
        data_np = to_numpy(data)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(data_np)
    if title:
        ax.set_title(title)
    fig.canvas.draw()
    plt.close()
    return fig

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def imshow(
    data: torch.Tensor,
    title: Optional[str]=None,
    origin: Literal["upper", "lower"]="upper"
) -> Figure:
    """
    Displays a 2D tensor as an image using matplotlib's imshow.

    Args:
        data (torch.Tensor): A 2D tensor to be visualized as an image.
        title (Optional[str]): Optional title for the plot.
        origin (Literal["upper", "lower"]): Determines the origin of the image.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the image plot.
    """
    if isinstance(data, torch.Tensor):
        data_np = to_numpy(data)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    im = ax.imshow(data_np, aspect="auto", origin=origin, interpolation='none')
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig

def hist(
    data: torch.Tensor,
    bins: int = 50,
    title: Optional[str] = None
) -> Figure:
    """
    Plots a histogram from a 1D tensor using matplotlib.

    Args:
        data (torch.Tensor): A 1D tensor representing the data to be plotted.
        bins (int): Number of histogram bins.
        title (Optional[str]): Optional title for the plot.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the histogram.
    """
    if isinstance(data, torch.Tensor):
        data_np = to_numpy(data)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(data_np, bins=bins)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.canvas.draw()
    plt.close()
    return fig


def heatmap_3d(
    data: torch.Tensor,
    anchor: int = 5,
    title: Optional[str] = None
) -> Figure:
    """
    Create a 3D heatmap plot of the given data.

    Args:
        data (torch.Tensor): The data to plot.
        anchor (int): The anchor point for the heatmap.
        title (Optional[str]): Optional title for the plot.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the heatmap.
    """

    data = data.permute(1,0)
    T, N = data.shape
    x = torch.arange(N)

    data_np = to_numpy(data)
    x = to_numpy(x)

    fig = plt.figure(figsize=(DEFAULT_FIGSIZE))
    ax = fig.add_subplot(111, projection='3d')

    for i, d in reversed(list(enumerate(data_np))):
        if i % (T//anchor) == 0:
            alpha = 1
        else:
            alpha = 0.2
        ax.plot(x, d, zs=i, zdir='y', color="blue", alpha=alpha, linewidth=1)

    ax.view_init(elev=20, azim=-60)
    xlim = ax.get_xlim3d() #type: ignore
    ylim = ax.get_ylim3d() #type: ignore
    zlim = ax.get_zlim3d() #type: ignore

    ax.set_box_aspect([1,2,1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_xlabel('Index')
    ax.set_ylabel('Frame')
    ax.set_zlabel('Value')
    if title is not None:
        ax.set_title(title)
    fig.canvas.draw()
    plt.close()
    return fig


def duration(
    duration: torch.Tensor,
    phonemes: List[str],
    spectrogram: torch.Tensor,
    title: Optional[str] = None
) -> Figure:

    boundaries = torch.cumsum(duration, dim=0)
    starts = torch.cat([torch.zeros(1, device=duration.device), boundaries[:-1]])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    xlabel = "Frames"
    extent = (0, spectrogram.shape[1], 0, spectrogram.shape[0])
    boundaries_plot = boundaries
    starts_plot = starts

    spectrogram_np = to_numpy(spectrogram)
    boundaries_plot = to_numpy(boundaries_plot)
    starts_plot = to_numpy(starts_plot)

    im = ax.imshow(spectrogram_np, aspect="auto", origin="lower",interpolation='none', extent=extent)

    # 5. 音素ラベルと境界線の描画
    y_max = extent[3]
    text_y_pos = y_max * 0.95  # 上部5%の位置に文字を置く

    for start, end, ph in zip(starts_plot, boundaries_plot, phonemes):
        if start < spectrogram.shape[1]:
            # 境界線 (白点線)
            ax.vlines(end, 0, y_max, colors='white', linestyles='dotted', alpha=0.7, linewidth=1)
    
            # 音素ラベル (区間の中央に配置)
            mid_point = float((start + end) / 2)
            ax.text(mid_point, text_y_pos, ph,
                    color='white', fontweight='bold', fontsize=12,
                    ha='center', va='top',
                    bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1)
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency bins")

    if title is not None:
        ax.set_title(title)
    fig.canvas.draw()
    plt.close()
    return fig
