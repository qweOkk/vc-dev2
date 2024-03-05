import matplotlib.pyplot as plt


def plot_spec(spec, title=None, figsize=(10, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(spec[0, 0, :, :].numpy())
    plt.title(title)
    plt.savefig('/home/wangli/ASGSR/model/' + title + '.jpg')
