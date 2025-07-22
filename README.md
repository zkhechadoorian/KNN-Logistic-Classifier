# **Knn Logistic Regression Classifier**

This project demonstrates the implementation of supervised learning algorithms using pure NumPy. It includes training, evaluation, and comparison of models on real-world datasets, showcasing an end-to-end approach to machine learning. The codebase is designed to be reusable and serves as a teaching and portfolio exercise.

# Project Directory Structure: kNN-Logistic-Regression-Classifier


```bash
├── ModelAnalysis.pdf                  # PDF report analyzing model performance
├── Models.ipynb                       # Jupyter Notebook for running and visualizing models
├── PythonFiles/                       # Contains Python modules for reusable code
│   ├── Functions.py                   # Helper functions used across the project
│   ├── Models.py                      # kNN and Logistic Regression model definitions
├── README.md                         # Project overview and instructions
├── data/                              # Collection of datasets used for training/testing
│   ├── Rice_Cammeo_Osmancik.arff.txt
│   ├── Short-agaricus-lepiota.data
│   ├── Short_Rice_Cammeo_Osmancik.arff.txt.txt
│   ├── Short_adult.data
│   ├── adult.data
│   ├── agaricus-lepiota.data
│   ├── car_evaluation.data
│   ├── ionosphere.data
│   └── iris.data
└── requirements.txt                  # List of dependencies for setting up the environment
```
## 🔧 Setup and Installation Instructions

1. Move the Zip file from the local machine to the remote server using `scp`
   ```bash
   scp /path/to/local/kNN-Logistic-Regression-Classifier.zip username@remote_host:/path/to/remote/directory
   ```
2. SSH into the remote server
    ```bash
    ssh -i /path/to/private/key username@remote_host
    ```
3. Unzip the file on the remote server
    ```bash
    unzip kNN-Logistic-Regression-Classifier.zip
    ```
4. Navigate to the project directory
    ```bash
    cd kNN-Logistic-Regression-Classifier
    ```
5. Set up the virtual environment
    ```bash
    python3.10 -m venv .venv
    source .venv/bin/activate    # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    pip install notebook
    ```
6. Set Up Remote Development

If you want to run the notebook on a remote server, follow these steps to set up VS Code for remote development:

#### Install Required Extensions and Tools

1. **Install Remote - SSH Extension**
  - Open VS Code on your local machine.
  - Click the Extensions view icon in the Activity Bar or press `Ctrl + Shift + X` / `⌘ + Shift + X`.
  - In the search box, type **Remote - SSH**.
  - Select **Remote - SSH** (publisher: Microsoft).
  - Click **Install** and wait for the installation to complete.

2. **Install Python Extension**
  - In the Extensions view, search for **Python**.
  - Select **Python** (publisher: Microsoft).
  - Click **Install** to add Python support to VS Code.

3. **Install Jupyter Extension**
  - In the Extensions view, search for **Jupyter**.
  - Select **Jupyter** (publisher: Microsoft).
  - Click **Install** to enable Jupyter Notebook support in VS Code.

#### Connect to a Remote Host
1. Open the Command Palette by pressing `Ctrl + Shift + P` / `⌘ + Shift + P`.
2. Begin typing **Remote‑SSH: Connect to Host…** and select it.
3. If you already have hosts in `~/.ssh/config`, pick one. Otherwise:
  - Choose **Add New SSH Host…** and follow the prompts:
    - Host: `username@ip_address` (e.g., `umair@203.0.113.25`).
    - Select the file (`~/.ssh/config`) where the host entry should be saved.
  - After saving, pick the new host from the list to connect.
#### Authenticate
- **Key-based Authentication**: VS Code will use your default private key or let you pick one.

  #### Troubleshooting Remote Development

  If you encounter issues while setting up remote development, here are some common solutions:

  1. **SSH Connection Fails**  
    - Verify the SSH key is correctly configured and matches the server's authorized keys.
    - Ensure the server's IP address and hostname are correct.

  2. **VS Code Remote Window Doesn't Load**  
    - Ensure the Remote - SSH extension is installed and up-to-date.
    - Restart VS Code and try reconnecting.
    - Check the logs in the Output panel (`Ctrl + Shift + U` / `⌘ + Shift + U`) for error messages.

  4. **Permission Issues**  
    - Ensure the project directory and files have appropriate permissions (`chmod -R 755 kNN-Logistic-Regression-Classifier`).

  For further assistance, refer to the [VS Code Remote Development Documentation](https://code.visualstudio.com/docs/remote/remote-overview).

Once connected, VS Code will reload in a Remote window, allowing you to work on the server as if it were local.

7. Optional Step 
    - If you want skip the environment setup and `Models.py` file, you can use the pre-written `run.sh` This is useful if you want to run the project without wirring commands manually.
    ```bash
    chmod +x run.sh
    ./run.sh
    ```


## 🛠️ Step-by-Step Guide

1. Open and Run the Notebook
    - Open the `Models.ipynb` file in VS Code.
    - Ensure the kernel is set to the `.venv` interpreter.
    - Run all cells or step through them interactively.


  The notebook `Models.ipynb` is designed to be self-contained and reproducible. 

  ### Notebook Workflow Overview

  The notebook provides a structured approach to training and evaluating machine learning models using the provided datasets. Below is a high-level summary of the workflow:

  1. **Module Imports and Dataset Loading**  
    - Import necessary libraries and load datasets for analysis and modeling.

  2. **Exploratory Data Analysis (EDA)**  
    - Perform initial data exploration to understand dataset characteristics and identify preprocessing needs.

  3. **Model Initialization and Cross-Validation**  
    - Set up machine learning models and evaluate their performance using cross-validation techniques.

  4. **Hyperparameter Tuning for kNN**  
    - Optimize the `k` parameter for the k-Nearest Neighbors algorithm to achieve the best performance.

  5. **Final Model Training and Evaluation**  
    - Train the selected models on the full dataset and evaluate their performance on test data.

  This workflow ensures a systematic approach to machine learning, from data preparation to model evaluation.

  ###  Additional Python Files and Their Purpose
  The project includes two key Python files in the `PythonFiles` directory that encapsulate reusable code for model training and data handling:

  | File               | What it **is**                                                                                                                                                           | What it **tries to achieve**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | **`Models.py`**    | *Algorithm layer* — a self-contained module that hand-implements classical supervised-learning algorithms (today: **Logistic Regression** and **k-Nearest-Neighbours**). | 1. **Expose clean, reusable classes** (`LogisticRegression`, `kNN`) whose public methods mirror those you’d find in scikit-learn (`fit`, `predict`, `score/k_fold`).<br>2. **Teach the maths** by writing every step in NumPy only—no high-level ML library—so a reader can trace code ↔ equations line by line.<br>3. **Stay dependency-free & portable**: the file has *zero* project-specific imports, so it can be dropped into any notebook or script without modification.<br>4. **Provide a CLI front-end** (via `argparse`) so you can train/evaluate a model from the shell with one command (`python Models.py --model knn --dataset rice …`).                             |
  | **`Functions.py`** | *Utility & data-engineering layer* — a grab-bag of helpers for loading, cleaning, analysing, and plotting the project’s tabular datasets.                                | 1. **Standardise data ingestion** with `readFile_*` functions that hide each UCI dataset’s quirks (delimiters, missing tokens, column names).<br>2. **Offer preprocessing shortcuts** (normalisation, one-hot encoding, simple train/test splits) so the notebook can call a single helper instead of rewriting boilerplate.<br>3. **Lightweight EDA & visualisation**: routines like `dataAnalysis()` and `bestKValue()` spit out summary stats or validation-accuracy curves in one line of code.<br>4. **Decouple notebooks from plumbing**: by isolating I/O and plotting here, `Models.ipynb` can stay focused on the narrative and experiments rather than messy housekeeping. |

  #### How They Fit Together

  1. **`Functions.py`** turns raw `.data`/`.txt` files into clean NumPy arrays (plus quick visuals) ➜
  2. **`Models.py`** consumes those arrays, trains from-scratch algorithms, and returns predictions/metrics ➜
  3. **`Models.ipynb`** orchestrates both, telling the story and comparing results.

  Think of it as **data prep (Functions)** → **learning logic (Models)** → **experiment notebook (IPYNB)**—a clean separation of concerns that keeps each part easy to test, reuse, and extend.

  ### CLI Usage to Run Models Independently from the Notebook (Optional)

  You can run the models directly from the command line using the following general form:

  ```bash
  python PythonFiles/Models.py --model {logistic|knn}          \
           --dataset {ionosphere|adult|rice|mushroom} \
           [--folds 5]                     \
           [--lr 0.01] [--iters 1000]      \
           [--k 5] [--test_split 0.2]
  ```

  #### Arguments:
  - `--model`: Specifies the algorithm to use. Choose between `logistic` (Logistic Regression) or `knn` (k-Nearest Neighbors).
  - `--dataset`: Selects the dataset for training/testing. Options include `ionosphere`, `adult`, `rice`, and `mushroom`.
  - `--folds`: (Optional) Number of folds for cross-validation. Default is 5.
  - `--lr`: (Optional) Learning rate for Logistic Regression. Default is `0.01`.
  - `--iters`: (Optional) Number of iterations for Logistic Regression. Default is `1000`.
  - `--k`: (Optional) Number of neighbors for kNN. Default is `5`.
  - `--test_split`: (Optional) Fraction of data to reserve for testing. Default is `0.2`.

  #### Example Usage:
  Note: The following commands assume you are in the project directory and have the necessary Python environment set up and it's activeted using the following command
  ```bash
  source .venv/bin/activate    # On Windows: .venv\Scripts\activate
  ```
1. Train a Logistic Regression model on the `adult` dataset:
     ```bash
     python PythonFiles/Models.py --model logistic --dataset adult --lr 0.01 --iters 1000
     ```

  This CLI interface provides flexibility for experimenting with different models, datasets, and hyperparameters directly from the terminal.

## 📊 Results

The Porject breifly summarizes the performance of the models on various datasets, including accuracy, precision, recall, and F1-score. The results are visualized in the notebook for easy comparison. We can see how we can use the kNN and Logistic Regression models to classify the datasets effectively and use the accuracy and other metrics to evaluate their performance.
