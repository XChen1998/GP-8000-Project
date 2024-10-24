import os
import matplotlib.pyplot as plt
import numpy as np


def read_performance_data_txt(file_name):
    training_data = []
    testing_data = []

    # Initialize empty lists to store each metric across all epochs
    training_metrics = {'Loss': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    testing_metrics = {'Loss': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

    # Read the TXT file
    with open(file_name, mode='r') as file:
        lines = file.readlines()

        # Skip the header and process each line
        for line in lines[1:]:  # Skip the header
            row = line.strip().split(",")
            phase = row[1].strip()
            metrics = list(map(float, row[2:]))  # Convert metric values to float

            # Assign metrics to either training or testing based on the phase
            if phase == 'Training':
                for i, key in enumerate(training_metrics):
                    training_metrics[key].append(metrics[i])
            elif phase == 'Testing':
                for i, key in enumerate(testing_metrics):
                    testing_metrics[key].append(metrics[i])

    # Convert the dictionary into a nested list
    training_data = [training_metrics[key] for key in training_metrics]
    testing_data = [testing_metrics[key] for key in testing_metrics]

    return training_data, testing_data


def plot_bar_comparison(layer_list):
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#AE48C8', '#F44040', '#15DB3F', '#E9AA18', '#2237d6']  # Different colors for each ResNet layer
    hatches = ['\\\\', '///', '++', "xxx", "---"]
    save_dir = "./plotting/optimal_testing_performance"

    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    testing_data_all = {}

    # Read data for each layer configuration
    for layer in layer_list:
        file_name = f"./ResNet-{layer}_TumorClassificationlogs/ResNet-{layer}_TumorClassificationexperiment_1_log.txt"
        _, testing_data = read_performance_data_txt(file_name)
        testing_data_all[layer] = testing_data

    # Extract the optimal performance (min/max) for each metric in the testing phase
    optimal_performance = {metric: [] for metric in metrics}

    for metric_idx, metric in enumerate(metrics):
        for layer in layer_list:
            optimal_performance[metric].append(min(testing_data_all[layer][metric_idx]) if metric == 'Loss' else max(
                testing_data_all[layer][metric_idx]))

    # Plot bar charts for each metric
    for metric in metrics:
        plt.figure(figsize=(7, 4.5))
        metric_values = optimal_performance[metric]
        x_labels = [str(layer) for layer in layer_list]  # Convert model layers to string for even distribution
        bars = plt.bar(x_labels, metric_values, color=[colors[i % len(colors)] for i in range(len(layer_list))],
                       hatch=[hatches[i % len(hatches)] for i in range(len(layer_list))], edgecolor='black', width=0.6,
                       label=[f'ResNet-{layer}' for layer in layer_list])

        # Add labels at the top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=14)

        plt.xlabel('ResNet Model Layer', fontsize=15)
        plt.ylabel(metric, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'Optimal_Testing_{metric}_comparison.png')

        plt.legend(bbox_to_anchor=(0.45, 1.3), loc='upper center', ncol=3, fontsize=16, columnspacing=0.5)
        plt.grid(True, axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

        plt.tight_layout()
        plt.subplots_adjust(left=0.12, right=0.98, top=0.8, bottom=0.15)

        plt.savefig(save_path)  # Save figure to the directory
        plt.close()  # Close the figure to free memory


# Example usage
layer_list = [18, 34, 50, 101, 152]  # List of ResNet layers

plot_bar_comparison(layer_list)
