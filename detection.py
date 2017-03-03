import matplotlib.image as mpimg
import numpy as np
import cv2
import copy
from functools import partial
from scipy import ndimage as ndi
from skimage.feature import hog

import config

DELTA_CENTER = config.detection['delta_center']
DELTA_HEIGHT_WEIGHT = config.detection['delta_height_width']
SMOOTHING_WEIGHT = config.detection['smoothing_weight']
SMOOTHING_WEIGHT_MOVE = config.detection['smoothing_weight_move']
CONFIDENCE_CEILING = config.detection['confidence_ceiling']


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=True,
               visualise=vis, feature_vector=feature_vec)


def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


def _transform_to_color_space(image, color_space):
    color_transformations = {
        'HSV': partial(cv2.cvtColor, code=cv2.COLOR_RGB2HSV),
        'LUV': partial(cv2.cvtColor, code=cv2.COLOR_RGB2LUV),
        'HLS': partial(cv2.cvtColor, code=cv2.COLOR_RGB2HLS),
        'YUV': partial(cv2.cvtColor, code=cv2.COLOR_RGB2YUV),
        'YCrCb': partial(cv2.cvtColor, code=cv2.COLOR_RGB2YCrCb)
    }
    return color_transformations.get(color_space, np.copy)(image)


def _get_hog_features(feature_image, hog_channel, orient, pix_per_cell, cell_per_block):
    if hog_channel == 'ALL':
        return np.ravel([
            get_hog_features(feature_image[:, :, channel],
                             orient, pix_per_cell, cell_per_block,
                             vis=False, feature_vec=True)
            for channel in range(feature_image.shape[2])
        ])
    else:
        return get_hog_features(feature_image[:, :, hog_channel], orient,
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)


def extract_features(images, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0):
    features = []
    for image in images:
        image_features = []
        image = mpimg.imread(image)
        feature_image = _transform_to_color_space(image, color_space)
        # spatial features
        image_features.append(bin_spatial(feature_image, size=spatial_size))
        # histogram features
        image_features.append(color_hist(feature_image, nbins=hist_bins))
        # hog features
        image_features.append(_get_hog_features(feature_image, hog_channel, orient, pix_per_cell, cell_per_block))
        # put them all together
        features.append(np.concatenate(image_features))
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)

    return on_windows


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    img_features = []
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins,
                                   bins_range=hist_range)
        img_features.append(hist_features)
    if hog_feat:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        img_features.append(hog_features)

    # Return list of feature vectors
    return np.concatenate(img_features)


def make_heatmap(windows, image_shape):
    background = np.zeros(image_shape[:2])
    for window in windows:
        background[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return background


def find_windows(image):
    windows = []
    threshold = 0
    image[image <= threshold] = 0
    labels = ndi.label(image)
    for index in range(1, labels[1] + 1):
        nonzero = (labels[0] == index).nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])
        bbox = ((np.min(nonzero_x), np.min(nonzero_y)),
                (np.max(nonzero_x), np.max(nonzero_y)))
        windows.append(bbox)
    return windows


def combine_boxes(windows, image_shape):
    target_windows = []
    if len(windows) > 0:
        image = make_heatmap(windows, image_shape)
        target_windows = find_windows(image)
    return target_windows


class Window:
    def __init__(self):
        self.score = []


# Adapted most code and logic for this part from:
# https://github.com/wonjunee/udacity-detecting-vehicles
def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _unpack_boxes(box):
    start_x, start_y = box[0]
    end_x, end_y = box[1]
    return start_x, start_y, end_x, end_y


def find_center(box):
    start_x, start_y, end_x, end_y = _unpack_boxes(box)
    return (start_x + end_x) / 2.0, (start_y + end_y) / 2.0


def find_radius(box):
    start_x, start_y, end_x, end_y = _unpack_boxes(box)
    return (end_x - start_x) / 2, (end_y - start_y) / 2


def find_center_box(boxes):
    result = []
    for box in boxes:
        center = find_center(box)
        width, height = find_radius(box)
        move = (0, 0)
        result.append((center, width, height, move, 1))
    return result


def boxes_close(old_center, new_center, old_width, new_width,
                old_height, new_height):
    return True if (distance(old_center, new_center) < DELTA_CENTER and
                    abs(old_width - new_width) < DELTA_HEIGHT_WEIGHT and
                    abs(old_height - new_height) < DELTA_HEIGHT_WEIGHT) else False


def _smoothing(new, old, index):
    return (new[index] + old[index] * SMOOTHING_WEIGHT) / (SMOOTHING_WEIGHT + 1)


def average_centers(new_center, old_center):
    return _smoothing(new_center, old_center, 0), _smoothing(new_center, old_center, 1)


def calculate_move(new_center, old_center, old_move):
    w = SMOOTHING_WEIGHT_MOVE
    return ((new_center[0] - old_center[0] + w * old_move[0]) / (w + 1),
            (new_center[1] - old_center[1] + w * old_move[1]) / (w + 1))


def add_center_move(center, move):
    return center[0] + move[0] / 5, center[1] + move[1] / 5


def add_center_box(new_boxes, old_boxes, confidence_ceiling=CONFIDENCE_CEILING):
    fresh_boxes = []
    temp_new_boxes = copy.copy(new_boxes)
    w = 3

    for old_box in old_boxes:
        old_center, old_width, old_height, old_move, old_score = old_box
        new_boxes = copy.copy(temp_new_boxes)
        if old_score > 10:
            add_score = 2
        else:
            add_score = 1
        found = False
        for new_box in new_boxes:
            new_center, new_width, new_height, new_move, new_score = new_box
            if boxes_close(old_center, new_center, old_width, new_width, old_height, new_height):
                fresh_box = [average_centers(new_center, old_center),
                             (new_width + w * old_width) / (w + 1),
                             (new_height + w * old_height) / (w + 1),
                             calculate_move(new_center, old_center, old_move),
                             min(confidence_ceiling, old_score + add_score)]
                temp_new_boxes.remove(new_box)
                found = True
                break
        if not found:
            fresh_box = [add_center_move(old_center, old_move), old_width, old_height, old_move, old_score - 1]
        fresh_boxes.append(fresh_box)

    fresh_boxes += temp_new_boxes
    temp_fresh_boxes = copy.copy(fresh_boxes)
    for box in fresh_boxes:
        if box[-1] <= 0:
            temp_fresh_boxes.remove(box)

    return temp_fresh_boxes


def average_boxes(target_windows, old_boxes, image_shape):
    target_boxes = find_center_box(target_windows)
    new_boxes = add_center_box(target_boxes, old_boxes)

    filtered_boxes = [new_box for new_box in new_boxes if new_box[-1] > 2]

    new_windows = [
        ((int(new_center[0] - new_width), int(new_center[1] - new_height)),
         (int(new_center[0] + new_width), int(new_center[1] + new_height)))
        for new_center, new_width, new_height, new_move, new_score in filtered_boxes
    ]

    heat_map = make_heatmap(new_windows, image_shape)
    if np.unique(heat_map)[-1] >= 2:
        labels = ndi.label(heat_map)[0]
        heat_map_2nd = np.zeros_like(heat_map)
        heat_map_2nd[heat_map >= 2] = 1
        labels_2 = ndi.label(heat_map_2nd)
        for index in range(1, labels_2[1] + 1):
            nonzero = (labels_2[0] == index).nonzero()
            num = labels[nonzero[0][0], nonzero[1][0]]
            labels[labels == num] = 0
        heat_map = labels + heat_map_2nd
        new_windows = find_windows(heat_map)

    return new_windows, new_boxes
