from pathlib import Path

import matplotlib.pyplot as plt



def create_plots_dir() -> Path:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def save_histogram(data_array, headers, column_index) -> Path:
    plots_dir = create_plots_dir()

    column_name = headers[column_index]
    column_data = data_array[:, column_index]

    output_path = plots_dir / f"{column_name}_hist.png"

    plt.figure()
    plt.hist(column_data, bins=5)
    plt.title(f"{column_name} Distribution")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_scatter_plot(data_array,headers,x_index,y_index) -> Path:
    plots_dir = create_plots_dir()

    x_name = headers[x_index]
    y_name = headers[y_index]

    x_data = data_array[:, x_index]
    y_data = data_array[:, y_index]

    output_path = plots_dir / f"{x_name}_vs_{y_name}.png"

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.title(f"{x_name} vs {y_name}")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path