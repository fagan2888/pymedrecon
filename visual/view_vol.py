""" 3D volume visualizer originally written by Juan Nunez-Iglesias.

Original article is at
https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
"""
import matplotlib.pyplot as plt


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def multi_slice_viewer(volume):
    """ A multi-slice viewer function. Input is a 3D numpy array. After running
        this function, type "plt.show()" to display the plot. Then, it is
        possible to scroll through it with the 'j' and 'k' keys.
    """
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    print('scroll through slices with j and k keys')
    fig.canvas.mpl_connect('key_press_event', process_key)
