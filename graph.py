import json
from math import log

from sacred import Experiment, observers
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
plt.rc('axes', labelsize=11)  # fontsize of the x and y labels
plt.rc('axes', titlesize=11)  # fontsize of the title

graph_grok_experiment = Experiment('graph_grok')
observer = observers.FileStorageObserver('results/graph_grok')
graph_grok_experiment.observers.append(observer)


@graph_grok_experiment.config
def config():
    run = 84
    log_scale = True
    show = True


@graph_grok_experiment.automain
def main(run, log_scale, show):
    with open(f"results/grok/{run}/metrics.json") as f:
        data = json.load(f)

    train_loss = data['train_loss']['values']
    train_acc = data['train_acc']['values']
    test_loss = data['test_loss']['values']
    test_acc = data['test_acc']['values']
    steps = data['train_loss']['steps']

    if log_scale:
        steps = [log(s, 10) for s in steps]

    plt.plot(steps, train_loss, color='red', label='train')
    plt.plot(steps, test_loss, color='green', label='val')

    plt.title('Modular Division (training on 50% of data)')  # TODO: get op and actual data frac...
    plt.xlabel('Optimization Steps')
    plt.ylabel('Loss')
    if log_scale:
        plt.xlim(0.75, 6.25)
        plt.xticks([1, 2, 3, 4, 5, 6], ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
    plt.legend(loc='upper left', facecolor='#EAEAF2')

    plt.grid(axis='both', color='white', linestyle='-')
    ax = plt.gca()
    ax.set_facecolor('#EAEAF2')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', length=0.0, pad=10)

    fig = plt.gcf()
    fig.set_size_inches(7.05, 4.85)

    plt.savefig(observer.dir + f'/{run}_loss')
    if show:
        plt.show()

    plt.plot(steps, train_acc, color='red', label='train')
    plt.plot(steps, test_acc, color='green', label='val')

    plt.title('Modular Division (training on 50% of data)')  # TODO: get op and actual data frac...
    plt.xlabel('Optimization Steps')
    plt.ylabel('Accuracy')
    if log_scale:
        plt.xlim(0.75, 6.25)
        plt.xticks([1, 2, 3, 4, 5, 6], ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 20, 40, 60, 80, 100])
    plt.legend(loc='upper left', facecolor='#EAEAF2')

    plt.grid(axis='both', color='white', linestyle='-')
    ax = plt.gca()
    ax.set_facecolor('#EAEAF2')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', length=0.0, pad=10)

    fig = plt.gcf()
    fig.set_size_inches(7.05, 4.85)

    plt.savefig(observer.dir + f'/{run}_accuracy')
    if show:
        plt.show()
