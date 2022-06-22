import matplotlib.pyplot as plt
import numpy as np
import math
from multiprocessing import Process, Manager, Pool
import os

from giftools import render


class FracMap(object):
    def __init__(self):
        pass
    def __call__(self, x, c):
        f = lambda x, c: x**2 + c
        return f(x, c)


my_frac = FracMap()


class Camera:
    def __init__(self, resolution=(250,250), frame=([-2,2],[-2,2]), center=(0,0)):
        self.resolution = resolution  # dimensions in pixels
        self.frame = frame  # tuple storing complex domain
        self.center = ((self.frame[0][0]+self.frame[0][1])/2,(self.frame[1][0]+self.frame[1][1])/2)  # tuple storing center of the frame
        self.xlen = self.frame[0][1] - self.frame[0][0]
        self.ylen = self.frame[1][1] - self.frame[1][0]
        self.depth = 25

        self.frame_init = frame  # stores original complex domain before any transformations
        self.xlen_init = self.xlen
        self.ylen_init = self.ylen

    def update_position(self, new_frame):  # changes camera frame directly, doesn't yet support change in resolution
        self.frame = new_frame
        self.xlen = self.frame[0][1] - self.frame[0][0]
        self.ylen = self.frame[1][1] - self.frame[1][0]
        self.center = ((self.frame[0][0]+self.frame[0][1])/2,(self.frame[1][0]+self.frame[1][1])/2)

    def recenter(self, new_center, reinit=False):  # shifts same camera frame to be centered around a new point
        self.center = new_center
        frame = ([self.center[0]-self.xlen/2, self.center[0]+self.xlen/2],
                 [self.center[1]-self.ylen/2, self.center[1]+self.ylen/2])
        self.frame = frame
        if reinit:
            self.frame_init = frame  # stores original complex domain before any transformations
            self.xlen_init = self.xlen
            self.ylen_init = self.ylen

    def zoom(self, frame_count, frame_current, zoom_factor, depth, depth_scale=1):
        self.depth = self.depth * depth_scale

        R_val = (self.xlen_init - math.exp(math.log(self.xlen_init)*(frame_count-frame_current)/frame_count+math.log(self.xlen_init*zoom_factor)*frame_current/frame_count))/2
        I_val = (self.ylen_init - math.exp(math.log(self.ylen_init)*(frame_count-frame_current)/frame_count+math.log(self.ylen_init*zoom_factor)*frame_current/frame_count))/2

        R_adjusted = [self.frame_init[0][0] + R_val, self.frame_init[0][1] - R_val]
        I_adjusted = [self.frame_init[1][0] + I_val, self.frame_init[1][1] - I_val]

        self.update_position((R_adjusted, I_adjusted))

    def capture_frame(self, approach="process", show_trace=True, iterations=25, frame_current=0):
        param = 0
        if frame_current == 0:
            self.depth = iterations
        if approach == "process":
            with Manager() as manager:

                data = manager.list(range(self.resolution[1]))

                cores = 20
                p = [
                    Process(target=renderStrips, args=(i, self, my_frac, math.ceil(self.depth), param, data, cores))
                    for i in range(cores)
                ]
                for p_i in p:
                    p_i.start()
                for p_i in p:
                    p_i.join()

                X = np.array(data)

        if approach == 'baby':
            X = np.array(fractal(self, my_frac, self.depth, param))

        export_figure_matplotlib(X, str(frame_current), 120, 1, False)

        if show_trace:
            print('frame: ' + str(frame_current + 1))
            print(str(self.frame[0][0])+','+str(self.frame[0][1]))
            print(str(self.frame[1][0])+','+str(self.frame[1][1]))
            print("depth: " + str(self.depth))
            print()


class Animation:  # camera path should be a path parameterized from 0 to 1 guiding the frame center
    def __init__(self, camera=Camera(), depth=25, depth_scale=1, camera_path=lambda t: (0,0), frame_count=1, frame_duration=1/24, zoom_factor=1):
        self.cam = camera
        self.path = camera_path
        self.fcount = frame_count
        self.duration = frame_duration
        self.zoomf = zoom_factor
        self.depth = depth
        self.depth_scale = depth_scale

    def animate(self, make_gif=True, display_trace=True):  # frame by frame moves center along the parameterized path and then applies zoom
        for frame in range(self.fcount):
            center = self.path(frame/(self.fcount-1))
            self.cam.recenter(center)
            self.cam.zoom(frame_count=self.fcount, frame_current=frame, zoom_factor=self.zoomf, depth=self.depth, depth_scale=self.depth_scale)
            self.cam.capture_frame(iterations=self.depth, frame_current=frame, show_trace=display_trace)

        if make_gif:
            directory = 'frames/'
            frames = []

            for filename in os.listdir(directory):
                frames.append((int(filename[:-4]), 'frames/' + filename))

            frames = sorted(frames)
            print(frames)

            render([y for (x, y) in frames], "test", frame_duration=self.duration)


def fractal(cam, iterable, iterations, time):

    row, col, i = 0, 0, 0

    pixels = [''] * cam.resolution[1]
    for k in range(cam.resolution[1]):
        pixels[k] = [''] * cam.resolution[0]

    for row in range(cam.resolution[0]):
        for col in range(cam.resolution[1]):

            # if row == math.floor(w/2) and col == math.floor(h/2):
            #    print("Halfway There ;)")

            cx = ((cam.xlen)/cam.resolution[0])*row+cam.frame[0][0]
            cy = ((-cam.ylen)/cam.resolution[1])*col+cam.frame[1][1]

            # old scaling function in case something breaks
            # cx = (row - 3*w/4)/(0.25*w)
            # cy = (col - h/2)/(0.25*h)

            c = complex(cx, cy)
            x = complex(0, 0)

            for i in range(iterations):
                if abs(x) > 4:
                    break

                x = iterable(x, c)

            color = i
            pixels[col][row] = color

    return pixels


def rational_julia_set(width, height, iterations, time):  # deprecated

    w, h = width, height
    row, col, i = 0, 0, 0

    pixels = np.arange(w*h, dtype=np.uint16).reshape(h, w)

    for row in range(w):
        for col in range(h):

            cx = ((R_range[1]-R_range[0])/R_scale)*row+R_range[0]
            cy = ((I_range[0]-I_range[1])/I_scale)*col+I_range[1]

            x = complex(cx, cy)
            c = complex(time, 0)  # Point to evaluate the julia set at

            for i in range(iterations):
                p = x**2+2*x+1
                q = x+6

                if (abs(x) > 10) or q == 0: break
                x = p/q + c

            color = i
            pixels[col, row] = color

    return pixels


def export_figure_matplotlib(arr, f_name, dpi=120, resize_fact=1, plt_show=False):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap='twilight_shifted')
    plt.savefig("frames/" + f_name, dpi=(dpi * resize_fact))
    if plt_show:
        plt.show()
    else:
        plt.close()


def renderStrips(process, cam, iterable, iterations, time, final_render, core_count):
    for row in range(cam.resolution[1]):
        if row % core_count == process:
            I_new = (cam.frame[1][1] - (row+1)*(cam.frame[1][1]-cam.frame[1][0])/cam.resolution[1], cam.frame[1][1] - row*(cam.frame[1][1]-cam.frame[1][0])/cam.resolution[1])

            cam_new = Camera(resolution=(cam.resolution[0],1), frame=(cam.frame[0], I_new))

            final_render[row] = fractal(cam_new, iterable, iterations, time)[0]
