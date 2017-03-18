class Every:

    def __init__(self, every, offset=0):
        self._every = every
        self._offset = offset
        self._last = None

    def __call__(self, timestep):
        timestep = max(0, timestep - self._offset)
        if self._last is None:
            self._last = timestep
        if timestep - self._last < self._every:
            return False
        self._last += self._every
        return True


class Decay:

    def __init__(self, start, stop, steps):
        self._start = start
        self._stop = stop
        self._steps = steps or 1
        assert self._start >= self._stop
        assert self._steps

    def __call__(self, timestep):
        progress = min(timestep, self._steps) / self._steps
        mixed = (1 - progress) * self._start + progress * self._stop
        return mixed


class Statistic:

    def __init__(self, template, every=10000):
        self._template = template
        self._every = every
        self._values = []

    def __call__(self, value):
        self._values.append(value)
        if len(self._values) > self._every:
            average = sum(self._values) / len(self._values)
            print(self._template.format(average))
            self._values = []
