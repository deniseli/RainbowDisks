# Uses the organic "growing circles" technique

import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from scipy import stats
from scipy import ndimage as ndi

from skimage import color
from skimage import feature

WIDTH = 100
HEIGHT = 100
SPACE = 0.25

class Circle():

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = float(r)
        self.alive = True
        self.growth = random.uniform(0.05, 0.2)

    def update(self):
        if not self.alive: return
        self.r += self.growth
        dist_from_c = sqrdist(self.x, WIDTH/2, self.y, HEIGHT/2) ** 0.5
        bound = max(WIDTH, HEIGHT) / 3
        max_r = (bound - dist_from_c) / bound * 5
        if self.r > max_r:
            self.alive = False
        if blocked(self):
            self.r -= self.growth
            self.alive = False

    def color_segmentation(self):
        hue = 1.5*((WIDTH-self.x) + self.y) / (WIDTH+HEIGHT)
        if self.r < 0.5:
            x = 0.5*(0.5 - self.r)
            hue += random.uniform(-x,x)
        hue = 1.0*int(hue * 7) / 7
        return hue

    def to_draw(self):
        rgb = colorsys.hsv_to_rgb(self.color_segmentation(), 0.99, 0.99)
        rgb = (rgb[0] * 256, rgb[1] * 256, rgb[2] * 256)
        hex = '#%02x%02x%02x' % rgb
        return '''<circle cx="%s" cy="%s" r="%s" stroke-width="0" fill="%s"/>\n'''%(self.y, self.x, self.r, hex)

def to_draw(circles, w, h):
    svg = '''<svg height="400px" viewBox="0 0 %s %s">\n'''%(w,h)
    for c in circles:
        svg += c.to_draw()
    svg += "</svg>"
    return "<html><body>" + svg + "</body></html>"

def grid_circles():
    circles = []
    for i in range(WIDTH/10):
        for j in range(HEIGHT/10):
            circles.append(Circle(i*10,j*10,1))
    return circles

def normal(maxn):
    n = np.random.normal(maxn/2, maxn/7)
    if n < 0: return 0
    if n > maxn: return maxn
    return n

def rand_little_circle():
    x = normal(WIDTH)
    y = normal(HEIGHT)
    r = 0.15
    while blocked(Circle(x,y,r+SPACE)):
        x = normal(WIDTH)
        y = normal(HEIGHT)
    return Circle(x,y,r)

def sqrdist(x0, x1, y0, y1):
    return (x1-x0)**2 + (y1-y0)**2

def collide(a, b):
    if sqrdist(a.x, b.x, a.y, b.y) > (a.r + b.r + SPACE)**2:
        return False
    return True

def blocked(circ):
    scale = 1.0*block_mask.shape[0] / WIDTH
    for i in range(int((circ.x - circ.r) * scale), min(int((circ.x + circ.r) * scale), block_mask.shape[0]-1)):
        for j in range(int((circ.y - circ.r) * scale), min(int((circ.y + circ.r) * scale), block_mask.shape[1]-1)):
            if block_mask[i,j]:
                return True
    return False

def collision_test(circles):
    for i in range(len(circles) - 1):
        for j in range(i+1, len(circles)):
            if not circles[i].alive and not circles[j].alive:
                continue
            if collide(circles[i], circles[j]):
                circles[i].alive = False
                circles[j].alive = False

def update(circles):
    circles.append(rand_little_circle())
    for i in range(len(circles) - 1):
        if collide(circles[i], circles[-1]):
            circles.pop()
            return
    collision_test(circles)
    #if not circles[-1].alive:
    #    circles.pop()
    for c in circles:
        c.update()

def gen_mask():
    mask = np.array((WIDTH, HEIGHT))
    im = color.rgb2gray(misc.imread('square.jpg'))
    l = np.max(im.shape)
    newshape = (int(1.0*im.shape[0]/l * WIDTH * 2), int(1.0*im.shape[1]/l * HEIGHT * 2))
    im = misc.imresize(im, newshape)
    f = np.vectorize(lambda x: x < 100)
    blocks = f(im)
    l = np.max(im.shape)
    lbig = l * 2
    mask = np.array([[False]*lbig]*lbig, dtype=bool)
    diff0 = lbig - blocks.shape[0]
    diff1 = lbig - blocks.shape[1]
    mask[diff0/2:diff0/2+blocks.shape[0], diff1/2:diff1/2+blocks.shape[1]] = blocks
    #plt.imshow(mask, cmap=plt.cm.gray)
    #plt.show()
    return mask

if __name__ == "__main__":
    circles = []
    w = 100
    h = 100
    global block_mask
    block_mask = gen_mask()
    while len(circles) < 1200:
        update(circles)
    print to_draw(circles, w, h)
