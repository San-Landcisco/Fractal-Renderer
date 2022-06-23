import matplotlib.pyplot as plt
import numpy as np
import math
from multiprocessing import Process, Manager, Pool, cpu_count
import os
from random import randint

from giftools import render
from fastfrac import fast_fractal


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

    def zoom(self, frame_count, frame_current, zoom_factor):
        R_val = self.xlen_init/2*(1-zoom_factor**(frame_current/frame_count))
        I_val = self.ylen_init/2*(1-zoom_factor**(frame_current/frame_count))

        R_adjusted = [self.frame_init[0][0] + R_val, self.frame_init[0][1] - R_val]
        I_adjusted = [self.frame_init[1][0] + I_val, self.frame_init[1][1] - I_val]

        self.update_position((R_adjusted, I_adjusted))

    def capture_frame(self, approach="process", show_trace=True, iterations=100, frame_current=0, prob=False, points=1000):
        param = 0
        if frame_current == 0:
            self.depth = iterations
        if approach == "process":
            with Manager() as manager:

                data = manager.list(range(self.resolution[1]))

                cores = cpu_count()
                p = [
                    Process(target=renderStrips, args=(i, self, math.ceil(self.depth), param, data, cores))
                    for i in range(cores)
                ]
                for p_i in p:
                    p_i.start()
                for p_i in p:
                    p_i.join()

                X = np.array(data)

        if approach == 'baby':
            X = np.array(fast_fractal(self.array(), iterations))

        export_figure_matplotlib(X, str(frame_current), 120, 1, False)

        if show_trace:
            print('frame: ' + str(frame_current + 1))
            print(str(self.frame[0][0])+','+str(self.frame[0][1]))
            print(str(self.frame[1][0])+','+str(self.frame[1][1]))
            print("depth: " + str(iterations))
            print()

    def array(self):
        return np.array([[complex(((self.xlen)/self.resolution[0])*row+self.frame[0][0],
                            ((-self.ylen)/self.resolution[1])*col+self.frame[1][1])
                            for row in range(self.resolution[0])]
                            for col in range(self.resolution[1])], dtype=np.complex_)


class Animation:  # camera path should be a path parameterized from 0 to 1 guiding the frame center
    def __init__(self, camera=Camera(), depth=25, depth_scale=1, camera_path=lambda t: (0, 0), frame_count=1, fps=24, zoom_factor=1):
        self.cam = camera
        self.path = camera_path
        self.fcount = frame_count
        self.fps = fps
        self.zoomf = zoom_factor
        self.depth = depth
        self.depth_scale = depth_scale

    def film(self, frame, display_trace=False):
        #center = self.path(frame/(self.fcount-1))
        #self.cam.recenter(center)
        self.cam.zoom(frame_count=self.fcount, frame_current=frame, zoom_factor=self.zoomf)
        self.cam.capture_frame(approach="baby", iterations=self.depth*(self.depth_scale**frame), frame_current=frame, show_trace=display_trace)

    def animate(self, mode="strips", make_gif=True, display_trace=True):  # frame by frame moves center along the parameterized path and then applies zoom
        if mode == "strips":
            for frame in range(self.fcount):
                center = self.path(frame/(self.fcount-1))
                self.cam.recenter(center)
                self.cam.zoom(frame_count=self.fcount, frame_current=frame, zoom_factor=self.zoomf, depth=self.depth, depth_scale=self.depth_scale)
                self.cam.capture_frame(approach="process", iterations=self.depth, frame_current=frame, show_trace=display_trace)
        elif mode == "frames":
            loop = math.floor(self.fcount / self.fps)
            for round in range(loop):
                pool = Pool(processes = cpu_count())
                results = pool.map(self.film, range(round*self.fps, (round+1)*self.fps))
                pool.close()
                pool.join()
            pool = Pool(processes = cpu_count())
            results = pool.map(self.film, range(loop*self.fps+1, self.fcount))
            pool.close()
            pool.join()
        if make_gif:
            directory = 'frames/'
            frames = []

            for filename in os.listdir(directory):
                frames.append((int(filename[:-4]), 'frames/' + filename))

            frames = sorted(frames)
            print(frames)

            render([y for (x, y) in frames], "test", frame_duration=1/self.fps)

        print("Finished.. Yay! :)")


def fractal(cam, iterable, iterations, time, probabilistic=False, samples=1000):
    if probabilistic:
        pixels = np.zeros(cam.resolution)

        r = np.random.uniform(low=0, high=2, size=samples)  # radius
        theta = np.random.uniform(low=0, high=2*np.pi, size=samples)  # angle

        a = np.sqrt(r) * np.cos(theta)
        b = np.sqrt(r) * np.sin(theta)

        for sample in range(samples):
            c = complex(a[sample], b[sample])
            x = complex(0, 0)
            #print(c)
            for i in range(iterations):
                if abs(x) > 2:
                    break

                x = iterable(x, c)

                n = math.floor((x.real-cam.frame[0][0])/cam.xlen*cam.resolution[0])
                m = math.floor(-(x.imag-cam.frame[1][1])/cam.ylen*cam.resolution[1])
                #print(x)
                #print(str(n) + "," + str(m))
                if m in range(cam.resolution[0]) and n in range(cam.resolution[1]):
                    pixels[n,m] += 1
                else:
                    break
        return pixels
    else:
        row, col, i = 0, 0, 0

        pixels = [''] * cam.resolution[1]
        for k in range(cam.resolution[1]):
            pixels[k] = [''] * cam.resolution[0]

        for row in range(cam.resolution[0]):
            for col in range(cam.resolution[1]):
                cx = ((cam.xlen)/cam.resolution[0])*row+cam.frame[0][0]
                cy = ((-cam.ylen)/cam.resolution[1])*col+cam.frame[1][1]

                c = complex(cx, cy)
                x = complex(0, 0)

                for i in range(iterations):
                    if abs(x) > 4:
                        break

                    x = iterable(x, c)

                color = i
                pixels[col][row] = color

        return pixels # deprecated


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
    ax.pcolormesh(arr, vmin=-1, vmax=1, cmap='twilight_shifted')
    ax.imshow(arr, cmap='twilight_shifted')
    #ax.imshow(arr, cmap='magma')
    plt.savefig("frames/" + f_name, dpi=(dpi * resize_fact))
    if plt_show:
        plt.show()
    else:
        plt.close()


def renderStrips(process, cam, iterations, time, final_render, core_count):
    for row in range(cam.resolution[1]):
        if row % core_count == process:
            I_new = (cam.frame[1][1] - (row+1)*(cam.frame[1][1]-cam.frame[1][0])/cam.resolution[1], cam.frame[1][1] - row*(cam.frame[1][1]-cam.frame[1][0])/cam.resolution[1])

            cam_new = Camera(resolution=(cam.resolution[0],1), frame=(cam.frame[0], I_new))

            final_render[row] = fast_fractal(cam_new.array(), iterations)[0]
