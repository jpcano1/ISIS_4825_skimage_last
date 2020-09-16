import requests
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

def download_content(url, filename="image.jpg", image=True):
    """
    Function that downloads content and opens the content
    if it's an image
    :param url: the url of the image
    :param filename: the filename of the image
    :return: the image loaded in memory
    """
    # We get the image address
    r = requests.get(url)

    # We open the image in "write-mode"
    with open(filename, "wb") as f:
        # We write the content to the file
        f.write(r.content)
    if image:
        return io.imread(filename)
    else:
        return

def imshow(img, title=None, cmap="gray", axis=False):
    """
    Simple function that shows the image
    :param img: The image to be shown
    :param title: Title of image
    :param cmap: Image color map
    :param axis: Boolean to know if plot uses axis.
    """
    # Plot Image
    plt.imshow(img, cmap=cmap)

    # Ask about the axis
    if not axis:
        plt.axis("off")

    # Ask about the title
    if title:
        plt.title(title)

def visualize(img, title=None, cmap="gray", figsize: tuple=None):
    """
    A more complex function to plot an image it has
    an account on the figsize
    :param img: the image to be shown
    :param title: the title of the image
    :param figsize: the size of the image
    :param cmap: Image Color map
    :type figsize: tuple
    """
    # Creates the figure
    fig: plt.Figure = plt.figure()

    # We set the color map to gray
    plt.gray()

    # Validate the figsize
    if figsize:
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
    # Shows image
    plt.imshow(img, cmap=cmap)

    # Show title
    if title:
        plt.title(title)
    plt.axis("off")

def visualize_subplot(imgs: list, titles: list, 
                    division: tuple, figsize: tuple=None, cmap="gray"):
    """
    An even more complex function to plot multiple images in one or
    two axis
    :param imgs: The images to be shown
    :param titles: The titles of each image
    :param division: The division of the plot
    :param cmap: Image Color Map
    :param figsize: the figsize of the entire plot
    """

    # We create the figure
    fig: plt.Figure = plt.figure(figsize=figsize)

    # Validate the figsize
    if figsize:
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

    # We make some assertions, the number of images and the number of titles
    # must be the same
    assert len(imgs) == len(titles), "La lista de imágenes y de títulos debe ser del mismo tamaño"

    # The division must have sense w.r.t. the number of images
    assert np.prod(division) >= len(imgs)

    # A loop to plot the images
    for index, title in enumerate(titles):
        ax: plt.Axes = fig.add_subplot(division[0], 
                            division[1], index+1)
        ax.imshow(imgs[index], cmap=cmap)
        ax.set_title(title)
        plt.axis("off")