def none_wrapper(self):
    return lambda *args: None


class Context:
    def __init__(self, init=none_wrapper, close=none_wrapper):
        self.init_func = init
        self.close_func = close

        self.has_init = True
        self.has_close = True

    def init(self):
        if not self.has_init:
            return
        self.has_init = False
        return self.init_func(self)

    def close(self):
        if not self.has_close:
            return
        self.has_close = False
        return self.close_func(self)


# Callback class with init, cb, close and shared context


class Callback:
    def __init__(self, cb, context=Context()):
        self.callback_func = cb(context)
        self.context = context

    def init(self):
        if self.context.has_init:
            return self.context.init()

    def callback(self, *args):
        return self.callback_func(*args)

    def close(self):
        if self.context.has_close:
            return self.context.close()
