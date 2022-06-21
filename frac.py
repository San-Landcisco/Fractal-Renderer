import matplotlib.pyplot as plt
import numpy as np
import math
from multiprocessing import Process, Manager, Pool
import os

from giftools import render


def animate_static(width, height, R_interval, I_interval): #template function for animating a transition with an unmoving frame
    frame_count = 1

    for frame_current in range(frame_count):

        param = 0

        image = mandelbrot_set(width, height, DEPTH, param, R_interval, I_interval)
        export_figure_matplotlib(image, label + str(frame_current), 120, 1, False)

    print("Finished.. Yay :)")
    return


def animate_zoom():
    #template for animating a zoom into center of frame
    #Subsequent frames are in the same aspect ratio

    frame_count = 10
    zoom_factor = 1/2**5

    DEPTH = 256
    DEPTH_SCALE = zoom_factor**(-1/(frame_count))

    R_init = R_range[1]-R_range[0]
    I_init = I_range[1]-I_range[0]

    for frame_current in range(frame_count):
        #Shifts the window, accomplishing #zoom_factor over #frame_count frames

        DEPTH = math.ceil(DEPTH * DEPTH_SCALE)

        R_range[0] = R_range[0] + (R_init/2)*(1/frame_count)*(1-zoom_factor)
        I_range[0] = I_range[0] + (I_init/2)*(1/frame_count)*(1-zoom_factor)

        R_range[1] = R_range[1] - (R_init/2)*(1/frame_count)*(1-zoom_factor)
        I_range[1] = I_range[1] - (I_init/2)*(1/frame_count)*(1-zoom_factor)

        print(str(R_range[0])+','+str(R_range[1]))
        print(str(I_range[0])+','+str(I_range[1]))
        print(DEPTH)
        print()

        param = 0 #potential frame dependent parameter

        image = mandelbrot_set(R_scale, I_scale, DEPTH, param)
        export_figure_matplotlib(image, label + str(frame_current), 120, 1, False)

    print("Finished.. Yay :)")
    return


def mandelbrot_set(width, height, iterations, time, R_interval, I_interval):

    row, col, i = 0, 0, 0

    pixels = [''] * height
    for k in range(height):
        pixels[k] = [''] * width

    for row in range(width):
        for col in range(height):

            #if row == math.floor(w/2) and col == math.floor(h/2):
            #    print("Halfway There ;)")

            cx = ((R_interval[1]-R_interval[0])/width)*row+R_interval[0]
            cy = ((I_interval[0]-I_interval[1])/height)*col+I_interval[1]

            #old scaling function in case something breaks
            #cx = (row - 3*w/4)/(0.25*w)
            #cy = (col - h/2)/(0.25*h)

            c = complex(cx, cy)
            x = complex(0,0)

            for i in range(iterations):
                if (abs(x) > 4): break
                #x = (complex(abs(x.real),abs(x.imag)))**4 + c #Burning Ship
                #x = (x.conjugate())**2+c #Tricorn / Mandelbar

                x = np.sin(x) + c

            color = i
            pixels[col][row] = color

    return pixels


def rational_julia_set(width, height, iterations, time):

    w, h = width, height
    row, col, i = 0, 0, 0

    pixels = np.arange(w*h, dtype=np.uint16).reshape(h, w)

    for row in range(w):
        for col in range(h):

            cx = ((R_range[1]-R_range[0])/R_scale)*row+R_range[0]
            cy = ((I_range[0]-I_range[1])/I_scale)*col+I_range[1]

            x = complex(cx, cy)
            c = complex(time,0) #Point to evaluate the julia set at

            for i in range(iterations):
                p = x**2+2*x+1
                q = x+6

                if (abs(x) > 10) or q == 0: break
                x = p/q + c

            color = i
            pixels[col,row] = color

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


def renderStrips(process, width, height, iterations, time, R_interval, I_interval, final_render):
    for row in range(height):
        if row % 8 == process:
            I_new = (I_interval[1] - (row+1)*(I_interval[1]-I_interval[0])/height, I_interval[1] - row*(I_interval[1]-I_interval[0])/height)

            final_render[row] = mandelbrot_set(width, 1, iterations, time, R_interval, I_new)[0]


def renderStrip(row): #broken
    width = R_scale
    height = I_scale
    iterations = DEPTH
    time = 0
    R_interval = R_range
    I_interval = I_range
    final_render = data

    I_new = (I_interval[1] - (row+1)*(I_interval[1]-I_interval[0])/height, I_interval[1] - row*(I_interval[1]-I_interval[0])/height)

    final_render[row] = mandelbrot_set(width, 1, iterations, time, R_interval, I_new)[0]


#pixels = mandelbrot_set(R_scale, I_scale, iterations=DEPTH, time=1)

#export_figure_matplotlib(pixels, "fractal", 120, 1, False)

#animate_static(R_scale, I_scale, R_range, I_range)

#animate_zoom()


if __name__ == '__main__':
    #Domain controls for the plot
    R_range = [-2,2] #Real interval
    I_range = [-2,2] #Complex interval

    R_scale = 1000   #Width number of pixels
    I_scale = abs(math.ceil(R_scale * (I_range[0]-I_range[1])/(R_range[1]-R_range[0])))

    display_trace = True
    make_gif = True
    approach = "process"

    frame_count = 50
    zoom_factor = 0.1

    param = 0

    DEPTH = 25 #controls number of iterations of the map
    DEPTH_SCALE = 1

    #Values for the zoom functions
    R_init = R_range[1]-R_range[0]
    I_init = I_range[1]-I_range[0]

    Del_R = R_init/2*(1-zoom_factor)
    Del_I = I_init/2*(1-zoom_factor)

    R_adjusted = [0,0]
    I_adjusted = [0,0]

    R_adjusted[0] = R_range[0] #mutable R_range for stages of zoom
    R_adjusted[1] = R_range[1]
    I_adjusted[0] = I_range[0] #mutable I_range for stages of zoom
    I_adjusted[1] = I_range[1]


    for frame_current in range(frame_count):
        #Shifts the window, accomplishing #zoom_factor over #frame_count frames

        DEPTH = math.ceil(DEPTH * DEPTH_SCALE)

        #R_range[0] = R_range[0] + (R_init/2)*(1/frame_count)*(1-zoom_factor)
        #I_range[0] = I_range[0] + (I_init/2)*(1/frame_count)*(1-zoom_factor)

        #R_range[1] = R_range[1] - (R_init/2)*(1/frame_count)*(1-zoom_factor)
        #I_range[1] = I_range[1] - (I_init/2)*(1/frame_count)*(1-zoom_factor)

        #R_val = (R_init - math.sqrt(R_init**2-4*Del_R*R_init*frame_current/frame_count+4*Del_R**2*frame_current/frame_count))/2
        #I_val = (I_init - math.sqrt(I_init**2-4*Del_I*I_init*frame_current/frame_count+4*Del_I**2*frame_current/frame_count))/2

        R_val = (R_init - math.exp(math.log(R_init)*(frame_count-frame_current)/frame_count+math.log(R_init*zoom_factor)*frame_current/frame_count))/2
        I_val = (I_init - math.exp(math.log(I_init)*(frame_count-frame_current)/frame_count+math.log(I_init*zoom_factor)*frame_current/frame_count))/2

        R_adjusted[0] = R_range[0] + R_val
        I_adjusted[0] = I_range[0] + I_val

        R_adjusted[1] = R_range[1] - R_val
        I_adjusted[1] = I_range[1] - I_val

        if approach == "pool":
            with Pool(processes = 8) as pool:
                pool.imap_unordered(renderStrip, range(I_scale))

        if approach == "process":
            with Manager() as manager:

                data = manager.list(range(I_scale))

                p0 = Process(target = renderStrips, args = (0, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p0.start()

                p1 = Process(target = renderStrips, args = (1, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p1.start()

                p2 = Process(target = renderStrips, args = (2, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p2.start()

                p3 = Process(target = renderStrips, args = (3, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p3.start()

                p4 = Process(target = renderStrips, args = (4, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p4.start()

                p5 = Process(target = renderStrips, args = (5, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p5.start()

                p6 = Process(target = renderStrips, args = (6, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p6.start()

                p7 = Process(target = renderStrips, args = (7, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data))
                p7.start()

                p0.join()
                p1.join()
                p2.join()
                p3.join()
                p4.join()
                p5.join()
                p6.join()
                p7.join()

                X = np.array(data)

        export_figure_matplotlib(X, str(frame_current), 120, 1, False)

        if display_trace == True:
            print('frame: ' + str(frame_current + 1))
            print(str(R_adjusted[0])+','+str(R_adjusted[1]))
            print(str(I_adjusted[0])+','+str(I_adjusted[1]))
            print("depth: " + str(DEPTH))
            print()

    if make_gif == True:
        directory = 'frames/'
        frames = []

        for filename in os.listdir(directory):
            frames.append((int(filename[:-4]),'frames/' + filename))

        frames = sorted(frames)
        print(frames)

        render([y for (x,y) in frames], "test", frame_duration = 1/24)
