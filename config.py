from glob import glob

data = dict(
    base_folder='../../../Downloads',
)

classifier = dict(
    color_space='YCrCb',  # RGB, HSV, LUV, HLS, YUV, YCrCb
    orient=8,
    pix_per_cell=8,
    cell_per_block=2,
    hog_channel='ALL',  # 0, 1, 2, or "ALL"
    spatial_size=(16, 16),
    hist_bins=32,
    test_size=0.2,
    vehicles=glob('{}/vehicles/*/*.png'.format(data['base_folder'])),
    non_vehicles=glob('{}/non-vehicles/*/*.png'.format(data['base_folder'])),
    pickle_features='features.pickle',
    pickle_model='model.pickle'
)

detection = dict(
    delta_center=5000,
    delta_height_width=50,
    smoothing_weight=2.0,
    smoothing_weight_move=6.0,
    confidence_ceiling=40
)

process = dict(
    window_parameters=[
        {'x_start_stop': [None, None], 'y_start_stop': [400, 500], 'xy_window': (96, 96), 'xy_overlap': (0.75, 0.75)},
        {'x_start_stop': [None, None], 'y_start_stop': [400, 500], 'xy_window': (144, 144), 'xy_overlap': (0.75, 0.75)},
        {'x_start_stop': [None, None], 'y_start_stop': [430, 550], 'xy_window': (192, 192), 'xy_overlap': (0.75, 0.75)},
        {'x_start_stop': [None, None], 'y_start_stop': [460, 580], 'xy_window': (192, 192), 'xy_overlap': (0.75, 0.75)},
    ],
    video_to_process='project_video.mp4',
    video_processed='project_video_processed.mp4'
)
