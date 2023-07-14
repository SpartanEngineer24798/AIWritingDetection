import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_data(input_folder):
    data_dict = {}
    for key in os.listdir(input_folder):
        path = os.path.join(input_folder, key)
        data = []
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r") as f:
                data.append(json.load(f))
        data_dict[key] = data
    return data_dict

def preprocess_data(df):
    num_features = len(df.columns) - 1
    column_names = [f"feature{i+1}" for i in range(num_features)] + ["key"]
    df.columns = column_names

    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    return df

def plot_2d_pca(finalDf, targets, colors, folder, basename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['key'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   color=color,
                   s=50)
    ax.legend(targets)
    ax.grid()

    os.makedirs(folder, exist_ok=True)

    counter = 1
    filename = f'{basename}.png'
    while os.path.exists(os.path.join(folder, filename)):
        counter += 1
        filename = f'{basename}_{counter}.png'

    fig.savefig(os.path.join(folder, filename))

    pca_output = finalDf[['principal component 1', 'principal component 2']].values
    np.savetxt(os.path.join(folder, f'{basename}_pca_output.csv'), pca_output, delimiter=',')

    explained_variance_ratios = pca.explained_variance_ratio_
    np.savetxt(os.path.join(folder, f'{basename}_explained_variance_ratios.csv'), explained_variance_ratios, delimiter=',')


def plot_3d_pca(finalDf, targets, colors, folder, basename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['key'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   finalDf.loc[indicesToKeep, 'principal component 3'],
                   color=color,
                   s=50)

    ax.legend(targets)
    ax.grid()

    os.makedirs(folder, exist_ok=True)

    counter = 1
    filename = f'{basename}.png'
    while os.path.exists(os.path.join(folder, filename)):
        counter += 1
        filename = f'{basename}_{counter}.png'

    fig.savefig(os.path.join(folder, filename))

    pca_output = finalDf[['principal component 1', 'principal component 2', 'principal component 3']].values
    np.savetxt(os.path.join(folder, f'{basename}_pca_output.csv'), pca_output, delimiter=',')

    explained_variance_ratios = pca.explained_variance_ratio_
    np.savetxt(os.path.join(folder, f'{basename}_explained_variance_ratios.csv'), explained_variance_ratios, delimiter=',')

def main(input_folder):
    data = read_data(input_folder)
    df = pd.DataFrame()

    for key, value in data.items():
        for sublist in value:
            row = sublist + [key]
            df = df.append(pd.Series(row), ignore_index=True)

    df = preprocess_data(df)

    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9']
    targets = ['university professor', 'primary school student', 'William Shakespeare', 'Steve Irwin', 'Sherlock Holmes', 'Queen Elizabeth', 'Winston Churchill', 'J.R.R. Tolkien', 'Forrest Gump', 'Human']
    colors = [(1, 0.4, 0.2, 0.5), (1, 0.2, 1, 0.5), (0.4, 1, 0.2, 0.5), (0.2, 1, 0.8, 0.5), (0.8, 0.2, 1, 0.5), (1, 0.8, 0.2, 0.5), (0.2, 0.4, 1, 0.5), (0.2, 0.8, 1, 0.5), (0.8, 0.2, 0.8, 0.5), (0, 0, 0, 0.3)]

    x = df.loc[:, features].values
    y = df.loc[:, ['key']].values

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[['key']]], axis=1)
    plot_2d_pca(finalDf, targets, colors, 'results/clustering', '2D_PCA_1.png')

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=[f'principal component {i+1}' for i in range(pca.n_components_)])
    finalDf = pd.concat([principalDf, df[['key']]], axis=1)
    plot_3d_pca(finalDf, targets, colors, 'results/clustering', '3D_PCA_1.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Path to the input folder")
    args = parser.parse_args()
    main(args.input_folder)