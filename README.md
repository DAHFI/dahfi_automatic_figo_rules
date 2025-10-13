# Dahfi Automatic FIGO Rules**

This repository provides an automated implementation of the **FIGO (International Federation of Gynecology and Obstetrics) guidelines** for the classification of **cardiotocographic (CTG) tracings**. Its main purpose is to facilitate the **early detection of intrapartum fetal hypoxia** by applying advanced **signal analysis algorithms** to fetal heart rate (FHR) and uterine contraction data.

The project aims to support both research and clinical decision-making by standardizing how CTG recordings are interpreted according to international criteria. By automating the FIGO classification process, it helps reduce subjectivity and inter-observer variability in fetal monitoring assessments.

### **Data Source**

The data used in the example notebook are obtained from a **publicly available database**, which can be freely accessed and downloaded from **[PhysioNet](https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/)**.
This dataset includes real intrapartum cardiotocographic recordings collected at the University Hospital in Brno (Czech Republic), making it a valuable resource for testing and validating algorithms related to fetal monitoring and signal processing.

### **Example of Use**

An example of how to use the provided code can be found in the notebook **`example_of_use.ipynb`**.
This notebook demonstrates:

* How to load and preprocess CTG data from the PhysioNet database.
* How to apply the implemented algorithms that follow FIGO guidelines.
* How to interpret the resulting classifications and visual outputs.

By following the steps in the example notebook, users can easily reproduce the analysis, adapt it to their own datasets, or integrate the methods into broader fetal monitoring research pipelines.

### **Environment Setup**
To ensure full reproducibility and easy setup of the required dependencies, this project provides an environment configuration file named **`environment.yml`**.This file contains all the necessary libraries (both Conda and pip packages) and their specific versions used during development and testing.

1. **Create the Conda Environment** <br>
First, make sure you have Anaconda or Miniconda installed on your system.Then, from the root directory of the repository, run:
conda env create -f environment.yml
This command will automatically create a new Conda environment with all the dependencies defined in the file.

2. **Activate the Environment** <br>
Once the environment is created, activate it with:
conda activate dahfi_figo_env



