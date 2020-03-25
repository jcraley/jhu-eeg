import pickle


def save_obj(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)
