import requests
from scipy import stats

from skimage import io
from skimage import morphology
from skimage import measure

import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
plt.style.use("ggplot")

import sys
from tqdm import tqdm

import numpy as np

def create_and_verify(path1, path2):
    full_path = os.path.join(path1, path2)
    exists = os.path.exists(full_path)
    if exists:
        return full_path
    else:
        raise FileNotFoundError("La ruta no existe")

def read_listdir(dir):
    listdir = os.listdir(dir)
    full_dirs = list()
    for d in listdir:
        full_dir = create_and_verify(dir, d)
        full_dirs.append(full_dir)
    return np.sort(full_dirs)

def download_content(url, chnksz=1000, filename:str="image.jpg", image=True, zip=False):
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
            f.write(pkg)
            bar.update(int(len(pkg)/chnksz))
        bar.close()
    
    if image:
        format_ = filename.split(".")[1]
        assert format_ in ["png", "jpg", "jpeg"], "Wrong file type"
        return io.imread(filename)
    elif zip:
        assert filename.endswith(".zip"), "Wrong filee type"
        with ZipFile(filename, "r") as zfile:
            print("\nExtrayendo Zip...")
            zfile.extractall()
            print("Eliminando Zip...")
            os.remove(filename)
    elif image and zip:
        raise Exception("Just one have to be chosen: Image or Zip")
    
    return

def distance(vec1: tuple, vec2: tuple):
    x = (vec1[1] - vec2[1])**2
    y = (vec1[0] - vec2[0])**2
    return np.sqrt(x + y)

def diameter(contour):
    longest = 0
    a, b = None, None
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            vec1 = contour[i]; vec2 = contour[j]
            dist = distance(vec1, vec2)
            if dist > longest:
                longest = dist
                a, b = vec1, vec2
    return a, b, longest

def perimeter(contour):
    perimeter = 0
    for i in range(len(contour) - 1):
        vec1 = contour[i]; vec2 = contour[i+1]
        perimeter += distance(vec1, vec2)
    return perimeter

def cart2pol(x, y):
    """
    Function that calculates the polar coordinates
    given cartesian coordinates.
    :param x: The x cartesian coordinate
    :param y: The y cartesian coordinate
    :return: the polar coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

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

def find_start_point(img):
    """
    Taken From: https://www.kaggle.com/mburger/freeman-chain-code-second-attempt
    Locates the starting point of the chain code.
    :param img: The binarized image where we are going to work
    :return: the starting point of contour finding
    """
    start_point = tuple()
    founded = False
    # We go for the rows
    for i, row in enumerate(img):
        # Then the columns
        for j, value in enumerate(row):
            # If it has value of one, it is the point
            if value == 1:
                start_point = (i, j)
                founded = True
                break
        # If we found, we break
        if founded:
            break
    return start_point

def skeletonize(segmented, elem=None):
    """
    Function that skeletonizes the binarized image
    with a structuring element
    :param segmented: The binarized image
    :param elem: The Structuring element
    :return: the skeletonized image.
    """
    skel = np.zeros_like(segmented)
    seg_copy = segmented.copy()

    # Get a square of 3x3
    if not elem:
        elem = morphology.square(3)
    
    while True:
        # Get the opening of the image
        opening = morphology.binary_opening(seg_copy, elem)
        
        # Substract open from the original image
        # We do this by performing a·(~b)
        temp = np.logical_and(seg_copy, np.logical_not(opening))

        # Erode the original image and refine the skeleton
        eroded = morphology.binary_erosion(seg_copy, elem)

        # We take te union of all with logical or
        skel = np.logical_or(skel, temp)
        seg_copy = eroded.copy()

        # If there are no white pixels left the image 
        # has been completely eroded, quit the loop
        if (seg_copy != 0).sum() == 0:
            break

    return skel

def rugosity(image):
    var = image.var()
    max = image.max() ** 2
    return 1 - (1 / (1 + var/max))

def moments(image, order=3):
    arr = image.flatten()
    return stats.moment(arr, moment=order)

def centroid(img):
    M = measure.moments(img, 1)
    return M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]

def find_contours(img, start_point=None):
    """
    Taken From: https://www.kaggle.com/mburger/freeman-chain-code-second-attempt
    """

    if not start_point:
        start_point = find_start_point(img)

    directions = [0,  1,  2,
                7,      3,
                6,  5,  4]
    dir2idx = dict(zip(directions, range(len(directions))))

    change_j = [-1,  0,  1, # x or columns
                -1,      1,
                -1,  0,  1]

    change_i = [-1, -1, -1, # y or rows
                0,      0,
                1,  1,  1]

    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0] + change_i[idx], 
                    start_point[1] + change_j[idx])
        if img[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break

    count = 0
    while curr_point != start_point:
        # figure direction to start search
        b_direction = (direction + 5) % 8 
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0] + change_i[idx], 
                        curr_point[1] + change_j[idx])
            if img[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        if count == 10000: break
        count += 1

    return np.array(border), np.array(chain), count

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
        
def scale(img, min, max, dtype="uint8"):
    """
    Función que escala los valores de una imagen entre un 
    rango definido
    :param img: la image que vamos a alterar
    :param min: el valor mínimo que queremos que tenga la imagen
    :param max: el valor máximo que queremos que tenga la imagen
    :param dtype: el tipo de datos que queremos que tenga la imagen
    :return: la imagen escalada
    """
    img_min = img.min()
    img_max = img.max()
    m = (max - min) / (img_max - img_min)
    return (m * (img - img_min) + min).astype(dtype)