# Legal Text Decoder (VITMMA19 Deep Learning Project)

---

## Project Information

- **Selected Topic:** Legal Text Decoder
- **Student Name:** Benedek Ágota, WFXBHI
- **Aiming for +1 Mark:** Yes

---

## Solution Description


This project addresses the problem of classifying the complexity/difficulty of Hungarian legal texts on a 1–5 ordinal scale. The solution is built as a full machine learning pipeline, including advanced data preparation, baseline and incremental model development, hyperparameter sweep, and a Gradio-based GUI for model serving.


### Key Features

- **Containerized Pipeline:** All steps (data prep, training, evaluation, inference) run in a Docker container.
- **Data Preparation:** Automated download and extraction of the raw dataset, consensus-based test set creation, and advanced preprocessing using NLTK for Hungarian stopword removal.
- **Baseline Model:** Logistic Regression on engineered features for reference.
- **Incremental Model Development:** LSTM-based classifier with hyperparameter sweep (using W&B) for optimal configuration.
- **Advanced Evaluation:** Includes confusion matrices, F1/accuracy/precision/recall, and consensus quality analysis.
- **ML as a Service:** After training, the final model is served via a Gradio web GUI for interactive predictions.
- **Comprehensive Logging:** All steps log to `log/run.log` with configuration, data stats, model details, training/validation metrics, and final evaluation.

---

## Extra Credit Justification

- **Incremental Model Development:** Extensive sweep-based hyperparameter optimization (see attached sweep log and image). For transparency and reproducibility, the original sweep-based training script (`src/02_train_with_sweep.py`) and the sweep run logs are included as evidence of incremental model development and hyperparameter optimization.
- **Advanced Data Preparation:** NLTK-based Hungarian stopword removal and overlap removal between splits. See the file or logs for evidence.
- **ML as a Service:** Gradio GUI for the final model (see attached screenshot).

---

### Hyperparameter Sweep

The hyperparameter sweep was performed using W&B and the script `src/02_train_with_sweep.py`. The sweep explored a range of model parameters and training settings, and the best configuration was automatically selected for the final training run.

![W&B sweep results summary](sweep.png)
*Figure: Example sweep results from W&B showing the optimization process.*

![Best parameters found by sweep](parameters.png)
*Figure: The best hyperparameters found during the sweep.*

![Full sweep run visualization](whole_sweep.png)
*Figure: Visualization of the full sweep run and parameter search space.*

- The best model and config are saved in `models/`.
- The pipeline runs the final training automatically with these best parameters.
- The original sweep script and sweep logs are provided as evidence of the incremental development process and parameter search.

---

### Gradio GUI

After training and evaluation, the container can launch a Gradio web interface for interactive legal text classification. The interface allows users to input Hungarian legal text and receive a predicted complexity/difficulty score (1–5) with class probabilities.

> **Note:** The Gradio service is disabled by default. To enable it, edit `src/run.sh` and uncomment the following lines:
> ```bash
> # echo "Launching Gradio interface..."
> # python3 src/05_serve.py
> ```

![Screenshot of the Gradio interface](GUI.png)

*Figure: The Gradio-based web interface for interactive legal text classification.*

- **Script:** `src/05_serve.py`
- **Pipeline Config:** `src/run.sh`
- **Interface:** Textbox input, top-5 class probabilities, example texts (see screenshot above).

---

## Data Preparation

The data preparation script (`src/01_data_processing.py`) performs the following:

1. **Download & Extraction:** Downloads the raw dataset from a provided SharePoint link and extracts it.
2. **Consensus Test Set:** Builds a consensus-based test set from annotation files, ensuring no overlap with training data.
3. **Preprocessing:** Applies NLTK-based Hungarian stopword removal to all texts.
4. **Splitting:** Splits the data into train/validation/test, removing any overlaps between splits.
5. **Saving:** Saves the processed HuggingFace `DatasetDict` to `data/final_split_dataset/`.
---

## Logging

All scripts use a unified logger (`src/utils.py`) that outputs to both stdout and `log/run.log`. The log includes:

- Hyperparameters and configuration
- Data loading and preprocessing confirmation
- Model architecture and parameter counts
- Training and validation metrics per epoch
- Final evaluation results (accuracy, F1, confusion matrix, etc.)
- Inference results

---

## Model Architecture

- **Baseline:** Multinomial Logistic Regression on engineered features (text length, avg. sentence and word length).
- **Main Model:** Bidirectional LSTM with embedding, sweep-optimized hyperparameters (see `models/best_sweep_model_config.json` for final values).

---



## Docker Instructions

This project is fully containerized and can be run using Docker or Docker Compose. Follow the steps below:

### 1. Build the Docker Image

Run this command in the root directory:

```sh
docker build -t dl-project .
```

### 2. Run the Project

#### Option A: Using Docker (Required for Submission)

- **Linux/macOS:**
  ```sh
  docker run --gpus all \
      -v "$(pwd):/work" \
      -p 7860:7860 \
      --env-file .env \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e PYTHONPATH=/work \
      --name LegalText \
      dl-project > log/run.log 2>&1
  ```
- **Windows PowerShell:**
  ```powershell
  docker run --gpus all \
      -v "${PWD}:/work" \
      -p 7860:7860 \
      --env-file .env \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e PYTHONPATH=/work \
      --name LegalText \
      dl-project > log/run.log 2>&1
  ```

After the pipeline completes, access the Gradio GUI at [http://localhost:7860](http://localhost:7860).

#### Option B: Using Docker Compose (Optional)

You can also use Docker Compose for convenience:

```sh
docker-compose up
```

This will build and launch the full pipeline (data preparation, training, evaluation, and Gradio GUI) in a single step. All necessary volumes and runtime options are handled by the included `docker-compose.yml` file.

---




## File Structure

```
src/
	01_data_processing.py   # Data download, cleaning, splitting, NLTK stopword removal
	02_train_with_sweep.py  # Sweep-based model training (W&B)
	02_train.py             # Final model training with best parameters
	03_consensus_eval.py    # Baseline and LSTM evaluation, metrics, confusion matrix
	04_inference.py         # Sample inference script
	05_serve.py             # Gradio GUI for model serving
	model.py                # LSTMClassifier, dataset, and loss definitions
	config.py               # Hyperparameters and paths configuration
	utils.py                # Logger setup
	run.sh                  # Pipeline runner script
models/                   # Saved models, configs, vocabularies
log/
	run.log                 # Full pipeline log
requirements.txt          # All dependencies (including torch, nltk, gradio, wandb, etc.)
Dockerfile                # Container build instructions
docker-compose.yml        # Docker Compose configuration
```

---

## Dependencies

All dependencies are listed in `requirements.txt`.

> **Note:** Some packages (e.g., `numpy`, `networkx`) do not have pinned versions because specific version constraints caused Docker build errors. The build uses the latest compatible versions.
