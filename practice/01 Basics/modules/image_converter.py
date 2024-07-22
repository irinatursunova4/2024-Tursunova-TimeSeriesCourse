import numpy as np
import pandas as pd
import math
import cv2
import imutils
from google.colab.patches import cv2_imshow


def _img_preprocess(img: np.ndarray) -> np.ndarray:
    """ 
    Preprocess the raw image: convert to grayscale, inverse, blur slightly, and threshold it
    
    Parameters
    ----------
    img : raw image
    
    Returns
    -------
    prep_img : image after preprocessing
    """

    # INSERT YOUR CODE

    return prep_img


def _get_contour(img: np.ndarray) -> np.ndarray:
    """ 
    Find the largest contour in the preprocessed image

    Parameters
    ----------
    img : preprocessed image
    
    Returns
    -------
    contour : object contour
    """

    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = [cnt for cnt in contours if cv2.contourArea(cnt) > 500][0]

    return contour


def _get_center(contour: np.ndarray) -> tuple[float, float]:
    """ 
    Compute the object center

    Parameters
    ----------
    contour : object contour 
    
    Returns
    -------
    coordinates of the object center
    """

    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    return (center_x, center_y)


def _find_nearest_idx(array: np.ndarray, value: int) -> int:
    """
    Find index of element that is the nearest to the defined value 

    Parameters
    ----------
    array : array of values 
    value : defined value
 
    Returns
    -------
    idx : index of element that is the nearest to the defined value 
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return idx


def _get_coordinates_at_angle(contour: np.ndarray, center: tuple[float, float], angle: int) -> np.ndarray:
    """ 
    Find one point on contour that are located at the angle 

    Parameters
    ----------
    contour : object contour 
    center : object center
    angle : angle 
 
    Returns
    -------
    coordinates of one point on the contour
    """

    angles = np.rad2deg(np.arctan2(*(center - contour).T))
    angles = np.where(angles < -90, angles + 450, angles + 90)
    found = np.rint(angles) == angle
    if np.any(found):
        return contour[found][0]
    else:
        idx = _find_nearest_idx(angles, angle)
        return contour[idx]


def _get_edge_coordinates(contour: np.ndarray, center: tuple[float, float], angle_step: int) -> list[np.ndarray]:
    """ 
    Find points on contour that are located from each other at the angle step 

    Parameters
    ----------
    contour : object contour 
    center : object center
    angle_step : angle step for finding the contour points 
 
    Returns
    -------
    edge_coordinates : coordinates of the object center
    """

    edge_coordinates = []
    for angle in range(0, 360, angle_step):
        pt = _get_coordinates_at_angle(contour, center, angle)
        if np.any(pt):
            edge_coordinates.append(pt)

    return edge_coordinates


def _img_show(img: np.ndarray, contour: np.ndarray, edge_coordinates: list[np.ndarray], center: tuple[float, float]) -> None:
    """ 
    Draw the raw image with contour, center of the shape on the image and rais from starting center

    Parameters
    ----------
    img : raw image
    contour : object contour 
    edge_coordinates : contour points 
    center : object center
    """

    cv2.drawContours(img, [contour], -1, (0, 255, 0), 6)
    cv2.circle(img, center, 7, (255, 255, 255), -1)
    cv2.putText(img, "center", (center[0]-20, center[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 6)
    for i in range(len(edge_coordinates)):
        cv2.drawContours(img, np.array([[center, edge_coordinates[i]]]), -1, (255, 0, 255), 4)

    cv2_imshow(imutils.resize(img, width=200))


def image2ts(img: np.ndarray, angle_step: int = 10, visualize: bool = False) -> np.ndarray:
    """ 
    Convert image to time series by angle-based method

    Parameters
    ----------
    img : input image
    angle_step : angle step for finding the contour points 
    visualize : visualize or not image with contours, center and rais from starting center
    
    Returns
    -------
    ts : time series representation
    """

    ts = []

    prep_img = _img_preprocess(img)
    contour = _get_contour(prep_img)
    center = _get_center(contour)
    edge_coordinates = _get_edge_coordinates(contour.squeeze(), center, angle_step)

    if (visualize):
        _img_show(img.copy(), contour, edge_coordinates, center)

    for coord in edge_coordinates:
        #dist = math.sqrt((coord[0] - center[0])**2 + (coord[1] - center[1])**2)
        dist = math.fabs(coord[0] - center[0]) + math.fabs(coord[1] - center[1])
        ts.append(dist)

    return np.array(ts)