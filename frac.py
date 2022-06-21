import matplotlib.pyplot as plt
import numpy as np
import math


def animate_static(width, height, R_interval, I_interval):  # template function for animating a transition with an unmoving frame
    frame_count = 1

    for frame_current in range(frame_count):

        param = 0

        image = mandelbrot_set(width, height, DEPTH, param, R_interval, I_interval)
        export_figure_matplotlib(image, label + str(frame_current), 120, 1, False)

    print("Finished.. Yay :)")
    return


def animate_zoom():
    # template for animating a zoom into center of frame
    # Subsequent frames are in the same aspect ratio

    frame_count = 10
    zoom_factor = 1/2**5

    DEPTH = 256
    DEPTH_SCALE = zoom_factor**(-1/(frame_count))

    R_init = R_range[1]-R_range[0]
    I_init = I_range[1]-I_range[0]

    for frame_current in range(frame_count):
        # Shifts the window, accomplishing #zoom_factor over #frame_count frames

        DEPTH = math.ceil(DEPTH * DEPTH_SCALE)

        R_range[0] = R_range[0] + (R_init/2)*(1/frame_count)*(1-zoom_factor)
        I_range[0] = I_range[0] + (I_init/2)*(1/frame_count)*(1-zoom_factor)

        R_range[1] = R_range[1] - (R_init/2)*(1/frame_count)*(1-zoom_factor)
        I_range[1] = I_range[1] - (I_init/2)*(1/frame_count)*(1-zoom_factor)

        print(str(R_range[0])+','+str(R_range[1]))
        print(str(I_range[0])+','+str(I_range[1]))
        print(DEPTH)
        print()

        param = 0  # potential frame dependent parameter

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

            # if row == math.floor(w/2) and col == math.floor(h/2):
            #    print("Halfway There ;)")

            cx = ((R_interval[1]-R_interval[0])/width)*row+R_interval[0]
            cy = ((I_interval[0]-I_interval[1])/height)*col+I_interval[1]

            # old scaling function in case something breaks
            # cx = (row - 3*w/4)/(0.25*w)
            # cy = (col - h/2)/(0.25*h)

            c = complex(cx, cy)
            x = complex(0,0)

            for i in range(iterations):
                if abs(x) > 4:
                    break
                # x = (complex(abs(x.real),abs(x.imag)))**4 + c #Burning Ship
                # x = (x.conjugate())**2+c #Tricorn / Mandelbar

                x = x**2 + c

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
            c = complex(time, 0)  # Point to evaluate the julia set at

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


def renderStrips(process, width, height, iterations, time, R_interval, I_interval, final_render, core_count):
    for row in range(height):
        if row % core_count == process:
            I_new = (I_interval[1] - (row+1)*(I_interval[1]-I_interval[0])/height, I_interval[1] - row*(I_interval[1]-I_interval[0])/height)

            final_render[row] = mandelbrot_set(width, 1, iterations, time, R_interval, I_new)[0]


def renderStrip(row):  # broken
    width = R_scale
    height = I_scale
    iterations = DEPTH
    time = 0
    R_interval = R_range
    I_interval = I_range
    final_render = data

    I_new = (I_interval[1] - (row+1)*(I_interval[1]-I_interval[0])/height, I_interval[1] - row*(I_interval[1]-I_interval[0])/height)

    final_render[row] = mandelbrot_set(width, 1, iterations, time, R_interval, I_new)[0]


# pixels = mandelbrot_set(R_scale, I_scale, iterations=DEPTH, time=1)

# export_figure_matplotlib(pixels, "fractal", 120, 1, False)

# animate_static(R_scale, I_scale, R_range, I_range)

# animate_zoom()
