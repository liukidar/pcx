import json
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_double_descent(result, file_prefix, label_color, background_color, transparent):

    sizes = sorted(
        [int(x.split("_")[-1])/1e3 for x in result["Train losses"].keys()])
    train_losses = OrderedDict(
        sorted(result["Train losses"].items(), key=lambda t: int(t[0].split("_")[-1])))
    val_losses = OrderedDict(
        sorted(result["Val losses"].items(), key=lambda t: int(t[0].split("_")[-1])))
    final_train_losses = [v[-1] for v in train_losses.values()]
    final_val_losses = [v[-1] for v in val_losses.values()]
    test_losses = list(result["Test losses"].values())
    test_losses = list(OrderedDict(sorted(result["Test losses"].items(
    ), key=lambda t: int(t[0].split("_")[-1]))).values())
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.set_facecolor(background_color)
    ax.semilogx(sizes, final_train_losses, ".-", label="Train", color="C0")
    ax.semilogx(sizes, test_losses, ".-", label="Test", color="C1")
    ax.set_xticks([], minor=True)
    ax.set_xticks([10, 40, 100, 300, 800], minor=False)
    ax.vlines(40e3/1e3, ymin=0, ymax=max(test_losses),
              linestyle="dotted", color=label_color, label="Interpolation")
    ax.tick_params(colors=label_color, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color(label_color)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(label_color)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    legend = ax.legend(frameon=False, fontsize=9)
    plt.setp(legend.get_texts(), color=label_color)
    plt.ylabel("Loss", color=label_color)
    plt.xlabel(r"N $(\times 10^3)$", color=label_color)
    fig.tight_layout()
    plt.savefig(file_prefix + "two_layer_double_descent.pdf",
                transparent=transparent)
    plt.show()


def plot_training(file_name):
    with open(file_name, "r") as fd:
        result = json.load(fd)

    for model_name in result["Train losses"].keys():
        plot_single_model(result, model_name)


def plot_single_model(result, model_name):
    train_losses = result["Train losses"][model_name]
    val_losses = result["Val losses"][model_name]
    fig, ax = plt.subplots()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title(model_name)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot two layer neural net result from experiments")

    parser.add_argument("result", type=argparse.FileType('r'),
                        help="the result json file.")
    parser.add_argument("--saveto", dest="file_prefix", type=str,
                        help="The file prefix to save the plots to", required=True)

    parser.add_argument("--bg", dest="bg", type=str,
                        help="The desired background color of the plot. Any acceptable matplotlib string (default \"black\")", default="black")
    parser.add_argument("--fg", dest="fg", type=str,
                        help="The desired label color of the plot. Any acceptable matplotlib string (default \"white\")", default="white")
    parser.add_argument("--transparent", action="store_true",
                        help="Save plot with transparent background", default=False)

    args = parser.parse_args()
    transparent = args.transparent
    bg = args.bg
    fg = args.fg
    result = json.load(args.result)
    plot_double_descent(result, args.file_prefix, fg, bg, transparent)
