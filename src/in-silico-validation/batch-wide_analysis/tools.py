import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize(data, folder, sufix=""):
    # Display summary statistics
    print(data.describe())

    # Histogram for 'size'
    plt.figure(figsize=(10, 6))
    sns.histplot(data['size'], kde=True)
    plt.title('Distribution of Size')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    # save the plot
    plt.savefig(folder + '\size_distribution2_' + sufix + '.png')
    plt.show()

    # Histogram for 'gc'
    plt.figure(figsize=(10, 6))
    sns.histplot(data['gc'], kde=True)
    plt.title('Distribution of GC Content')
    plt.xlabel('GC Content')
    plt.ylabel('Frequency')
    plt.savefig(folder + '\gc_distribution_' + sufix + '.png')
    plt.show()


    # Scatter plot of 'size' vs 'gc'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='size', y='gc', data=data)
    plt.title('Size vs GC Content')
    plt.xlabel('Size')
    plt.ylabel('GC Content')
    plt.savefig(folder + '\size_vs_gc_' + sufix + '.png')
    plt.show()

    # frequncy plot for rep_type, but dont count the values with value ´-´
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rep_type(s)', data=data[data['rep_type(s)'] != '-'])
    plt.title('Repeat Type')
    plt.xlabel('Repeat Type')
    plt.ylabel('Count')
    plt.savefig(folder + r'\repeat_type' + sufix + '.png')
    plt.show()

    # Box plot for 'size' vs 'rep_type'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='rep_type(s)', y='size', data=data)
    plt.title('Size vs Repeat Type')
    plt.xlabel('Repeat Type')
    plt.ylabel('Size')
    plt.savefig(folder + '\size_vs_repeat_type_' + sufix + '.png')
    plt.show()

    # frequency plot of orit_type(s)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='orit_type(s)', data=data[data['orit_type(s)'] != '-'])
    plt.title('Origin Type')
    plt.xlabel('Origin Type')
    plt.ylabel('Count')
    plt.savefig(folder + '\origin_type'+ sufix +'.png')
    plt.show()