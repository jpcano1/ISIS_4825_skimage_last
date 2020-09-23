import requests
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

import sys
import os

import zipfile
from tqdm import tqdm

def download_content(url, chnksz=1000, filename="image.jpg", image=True):
    """
    Función que se encarga de descargar un archivo deseado
    :param url: la url de descarga
    :param chnksz: Tamaño del chunk
    :param filename: El nombre del archivo
    :param image: Boolean que indica si lo que se descarga 
    es una imagen
    """
    # Se hace una petición de tipo GET
    try:
        r = requests.get(url, stream=True)

    # Si hay un problema, se cierra la ejecución.
    except Exception as e:
        print(f"Error de conexión con el servidor + {e}")
        sys.exit()

    # Se Abre el archivo en modo lectura
    with open(filename, "wb") as f:
        bar = tqdm(
            unit="KB",
            desc="Descargando archivos",
            total=int(
                np.ceil(int(r.headers["content-length"])/chnksz)
            )
        )

        # Para cada chunk del archivo, lo almaceno
        # y actualizo la pantalla
        for pkg in r.iter_content(chunk_size=chnksz):
            f.write(r.content)
            bar.update(int(len(pkg)/chnksz))
        bar.close()
    
    if image:
        return io.imread(filename)
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