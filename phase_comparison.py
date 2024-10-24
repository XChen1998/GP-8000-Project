import os

import matplotlib.pyplot as plt


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


import matplotlib.pyplot as plt
import os


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


def plot_performance_comparison(layer_list):
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#AE48C8', '#F44040', '#15DB3F', '#E9AA18', '#2237d6']
    # Initialize lists to hold performance data for all models
    training_data_all = {}
    testing_data_all = {}
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    save_dir = "./plotting/phase_comparison"

    # Read data for each layer configuration
    for layer in layer_list:
        file_name = f"./ResNet-{layer}_TumorClassificationlogs/ResNet-{layer}_TumorClassificationexperiment_1_log.txt"
        training_data, testing_data = read_performance_data_txt(file_name)
        training_data_all[layer] = training_data
        testing_data_all[layer] = testing_data

    # Plot training performance
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(7, 4.5))
        for j in range(len(layer_list)):  # Use 'j' for the layer_list loop to avoid overwriting 'i'
            layer = layer_list[j]
            plt.plot(range(len(training_data_all[layer][i])), training_data_all[layer][i],
                     marker=markers[j], color=colors[j], label=f'ResNet-{layer}', markersize=8,
                     markeredgecolor='black', markeredgewidth=1.5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Phase Index (Epochs)', fontsize=18)
        plt.ylabel(metric, fontsize=18)

        plt.legend(bbox_to_anchor=(0.45, 1.3), loc='upper center', ncol=3, fontsize=16, columnspacing=0.5)
        plt.grid(True, axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

        plt.tight_layout()
        plt.subplots_adjust(left=0.12, right=0.98, top=0.8, bottom=0.15)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'Training_{metric}.png')
        plt.savefig(save_path)  # Save figure to the directory
        plt.close()  # Close the figure to free memory


    # Plot testing performance
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(7, 4.5))
        for j in range(len(layer_list)):  # Use 'j' for the layer_list loop to avoid overwriting 'i'
            layer = layer_list[j]
            plt.plot(range(len(testing_data_all[layer][i])), testing_data_all[layer][i],
                     marker=markers[j], color=colors[j], label=f'ResNet-{layer}', markersize=8,
                     markeredgecolor='black', markeredgewidth=1.5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Phase Index (Epochs)', fontsize=18)
        plt.ylabel(metric, fontsize=18)
        plt.legend(bbox_to_anchor=(0.45, 1.3), loc='upper center', ncol=3, fontsize=16, columnspacing=0.5)
        plt.grid(True, axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

        plt.tight_layout()
        plt.subplots_adjust(left=0.12, right=0.98, top=0.8, bottom=0.15)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'Testing_{metric}.png')
        plt.savefig(save_path)  # Save figure to the directory
        plt.close()  # Close the figure to free memory


# Example usage
layer_list = [18, 34, 50, 101, 152]  # List of ResNet layers

plot_performance_comparison(layer_list)
