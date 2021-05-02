def none_wrapper(self):
    return lambda *args: None

# Callback class with init, cb, close and shared context


class Callback:
    def __init__(self, cb, init=none_wrapper, close=none_wrapper):
        self.init_func = init
        self.callback_func = cb(self)
        self.close_func = close

    def init(self):
        return self.init_func(self)

    def callback(self, *args):
        return self.callback_func(*args)

    def close(self):
        return self.close_func(self)
