import matplotlib.pyplot as plt


def plot_data(x, y, xlabel='range', ylabel='auc', title=''):
    # with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.plot(x, y, '*-', alpha=0.9)
    # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

    # plt.xlim([0.0, 1.0])
    plt.ylim([0., 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xticks(x)
    # plt.yticks(y)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()
