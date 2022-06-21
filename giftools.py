import imageio

def render(images, label, frame_duration = 0.5, path = ''): #takes in a list of images
    with imageio.get_writer(path + str(label) + '.gif', mode='I', duration = frame_duration) as writer:
        for image in images:
            frame = imageio.imread(image)
            writer.append_data(frame)

'''
def render(images, label, frame_duration = 0.5):
    frames = []
    for image in images:
        frames.append(imageio.imread(image))
        imageio.mimsave('/path/to/movie.gif', frames)
'''
