class sum_area_table(object):
    def __init__(self, size, data):
        width, height = size
        assert width * height == len(data)
        self.size = size
        self.data = data
        self.memo = [None for _ in xrange(width * height)]
        self.generate()
    def get(self, x, y):
        width, height = self.size
        index = y * width + x
        if (x < 0 or y < 0):
            return 0
        elif self.memo[index] is not None:
            return self.memo[index]
        else:
            cummulative = self.get(x - 1, y) + self.get(x, y - 1) - self.get(x - 1, y - 1) + self.data[index]
            self.memo[index] = cummulative
            return cummulative
    def total(self, x0, y0, x1, y1):
        a = self.get(x0 - 1, y0 - 1)
        b = self.get(x0 - 1, y1)
        c = self.get(x1, y0 - 1)
        d = self.get(x1, y1)
        return d - b - c + a
    def generate(self):
        width, height = self.size
        self.memo = [self.get(x, y) for y in xrange(height) for x in xrange(width)]
def example():
    w = h = 10 # or whatever parameters you want
    haar = sum_area_table(size=(w, h), data=range(w * h))
    print
    haar.total(2, 2, 4, 4)
    print
    sum((y * w + x for y in xrange(h) for x in xrange(w) if 2 <= y <= 4 and 2 <= x <= 4))

if __name__ == '__main__':
    example()