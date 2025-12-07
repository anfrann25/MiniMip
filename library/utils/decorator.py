import functools
import numpy as np

def cache_result(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(self, array, *args, **kwargs):
        # Create unique hashable key for caching
        if isinstance(array, np.ndarray):
            key_array = array.tobytes()
        else:
            key_array = str(array)

        # Create final key
        key = (func.__name__, key_array, args, tuple(kwargs.items()))

        # Return cached result if available
        if key in cache:
            return cache[key]

        # Otherwise compute
        result = func(self, array, *args, **kwargs)

        # Store in cache
        cache[key] = result

        return result

    return wrapper
