# AIWritingDetection

This repository contains research on AI Writing Detection conducted in collaboration with MoMA Labs.

## Stylometry Based MLP

The Stylometry Based MLP is a project inspired by GLTR and PAN stylometry efforts [https://pan.webis.de/] aimed at detecting split authorship in a given text. This work simplifies their research and focuses on extracting significant stylometric information from AI-generated text and comparing it with human-generated text.

## How to Use

To use this repository, follow the steps below:

1. Create a folder named "input" in the root directory of the project.
2. Within the "input" folder, create a subfolder for each category you want to analyze (e.g., "human" and "ai").
3. Place the text files to be analyzed in their respective subfolders. Each text file should be in the format of a ".txt" file.
   - Example structure:
   
     ```
     input
     ├── human
     │   ├── text1.txt
     │   └── text2.txt
     └── ai
         ├── text1.txt
         └── text2.txt
     ```
4. The program will automatically create an "output" folder in the root directory. This folder will contain the extracted features for each text file.
5. Additionally, a "results" folder will be created in the root directory. This folder will store plot images and the MLP checkpoint.
6. To run the main program, execute the following command in the terminal:

   ```
   python3 main.py --i input_directory --o output_directory --r results_directory
   ```

   Replace `input_directory` with the path to the "input" folder, `output_directory` with the desired path for the output folder, and `results_directory` with the desired path for the results folder.

By following these steps, you will be able to analyze the text files, extract stylometric features, and obtain results using the Stylometry Based MLP.