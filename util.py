import pickle as pickle_module


def log(string_to_log, logger=print):
    logger(string_to_log)


def pickle(object_to_pickle, destination):
    with open(destination, 'wb') as file:
        pickle_module.dump(object_to_pickle, file)


def unpickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle_module.load(f)
