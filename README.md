# Steel Surface Defect Classification and Cross-Dataset Generalization

Use a ResNet-50 deep learning model to classify steel surface defects on the NEU dataset and test how well the model generalizes to a different dataset (GC10).

This project was done as part of the *Advanced Foundations for Machine Learning* course.

---

## Project description

The project has two main objectives:

- **1. Supervised defect classification (NEU)**  
  Train and fine-tune a ResNet-50 CNN to classify six types of surface defects on hot-rolled steel strips using the **NEU Metal Surface Defects** dataset.

- **2. Cross-dataset generalization (NEU → GC10)**  
  Without retraining, evaluate the NEU-trained model on a second dataset, **GC10 Defects location for metal surface**, focusing on the common defect type *Inclusion*.  
  This highlights how a model that is perfect on one dataset can still struggle on data from a different plant/camera.

The whole pipeline is implemented in a single Google Colab notebook using TensorFlow / Keras.

---

## Datasets

### NEU Metal Surface Defects (NEU-DET)

- 6 classes: **Crazing, Inclusion, Patches, Pitted, Rolled, Scratches**  
- 300 grayscale images per class (200×200 px)  
- In this project the data are organized into:
  - **Train:** 276 images per class  
  - **Validation:** 12 images per class  
  - **Test:** 12 images per class  

Used as the **main training and evaluation dataset**.

> Kaggle link:  
> https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

---

### GC10 – Defects Location for Metal Surface (GC10-DET)

- 10 classes: **crease, crescent_gap, inclusion, oil_spot, punching_hole, rolled_pit, silk_spot, waist_folding, water_spot, welding_line**
- Larger grayscale images of steel strips with more complex backgrounds and different imaging conditions.
- In this project we only use the **`inclusion`** folder as an **external test set**.  
  The NEU-trained model is applied directly to GC10 inclusion images (no retraining).

> Kaggle link:  
> https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location

Raw datasets are **not** stored in this repository.  
Please download them from the original sources above.

---

## Methodology

### 1. Preprocessing

- Load images from NEU (train/valid/test) and GC10 (inclusion only).
- Resize all images to **224×224**.
- Convert to 3 channels if needed (NEU images are originally grayscale).
- Apply **ResNet-50 preprocessing** (`tf.keras.applications.resnet50.preprocess_input`).
- For NEU **training** images only:
  - Random rotations (±10°)  
  - Width / height shifts (up to 10%)  
  - Horizontal flips  

### 2. Model: ResNet-50 Transfer Learning

- Base model: `tf.keras.applications.ResNet50` with **ImageNet** weights, `include_top=False`.
- Top (custom head):
  - Global Average Pooling  
  - Dense(512, activation="relu")  
  - Dropout(0.5)  
  - Dense(6, activation="softmax")   ← 6 NEU defect classes

### 3. Training strategy

**Stage 1 – Train classifier head**

- Freeze all ResNet-50 layers.
- Train only the new dense layers.
- Optimizer: Adam, learning rate `1e-3`.
- Loss: categorical cross-entropy.
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau based on validation accuracy.

**Stage 2 – Fine-tune backbone**

- Unfreeze the top part of ResNet-50 (e.g., last few blocks).
- Train the full model (backbone + head) with smaller learning rate `1e-4`.
- Same callbacks as Stage 1.

The best model on the **NEU validation set** is saved and used for all evaluations.

---

## Results

### NEU Metal Surface Defects

- **Validation accuracy:** 100.0 %  
- **Test accuracy:** 100.0 %

Confusion matrices for both validation and test sets are perfect diagonals:  
all six classes (**Crazing, Inclusion, Patches, Pitted, Rolled, Scratches**) are predicted correctly for every image.

(See `figures/neu_confusion_matrix_val.png` and `figures/neu_confusion_matrix_test.png`.)

### Cross-dataset generalization: NEU → GC10

- Dataset: GC10 **inclusion** images (216 samples).
- Evaluation: count how many images are predicted as the NEU **“Inclusion”** class by the NEU-trained model.

Results:

- **Correct predictions:** 85 / 216  
- **Accuracy on GC10 inclusion:** **39.4 %**

This is much lower than the 100% accuracy on NEU and clearly shows the effect of **domain shift** between datasets (different cameras, lighting, background texture, and defect appearance).

Example predictions (correct vs misclassified) are shown in  
`figures/gc10_inclusion_predictions.png`.

---

## Repository structure

Suggested layout (adapt to your repo):

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Steel_Surface_Defect_Classification.ipynb
└── figures/
    ├── training_curves.png
    ├── neu_confusion_matrix_val.png
    ├── neu_confusion_matrix_test.png
    └── gc10_inclusion_predictions.png
