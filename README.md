# Palm Recognition

Classifies palm photographs of four people with convolutional neural networks in Keras. The full experiment lives in `Untitled1.ipynb` (Shady Nashed, 24822832).

Images are resized to 180×180, augmented on the fly, and trained as a four-class problem: one class per person. Custom CNNs are trained from scratch, then a frozen ImageNet VGG16 backbone is fine-tuned with a new classification head.

## Dataset

Images sit in a `Palm` folder next to the notebook. Each split is a directory of class folders, which is what `image_dataset_from_directory` expects.

```
Palm/
├── TrainSet/
│   ├── MyPalmTrainSet/
│   ├── P2TrainSet/
│   ├── P3TrainSet/
│   └── P4TrainSet/
├── ValidationSet/
│   ├── MyPalmValidationSet/
│   ├── P2ValidationSet/
│   ├── P3ValidationSet/
│   └── P4ValidationSet/
└── TestSet/
    ├── MyPalmTestSet/
    ├── P2TestSet/
    ├── P3TestSet/
    └── P4TestSet/
```

Counts recorded in the notebook:

| Split | Images | Classes |
| --- | ---: | ---: |
| Train | 174 | 4 |
| Validation | 40 | 4 |
| Test | 20 | 4 |

Training includes 43 images of the author's palm and 49 of person 2. Validation is balanced at 10 images per person.

## Pipeline

1. Load each split with `image_dataset_from_directory` at `image_size=(180, 180)` and `batch_size=32`.
2. Augment training images with horizontal flip, rotation of 0.2 (about 72°), and zoom of 0.2.
3. Scale pixels with `Rescaling(1/255)` for the custom CNNs. The VGG16 model uses `keras.applications.vgg16.preprocess_input` instead.
4. Train with RMSprop and `sparse_categorical_crossentropy`, tracking accuracy.
5. Save the weights with the lowest validation loss via `ModelCheckpoint`.
6. Reload that checkpoint and evaluate it on the test set. The first model also plots training curves and a validation confusion matrix.

## Models

Every custom network is a stack of `Conv2D` (ReLU, 3×3) and `MaxPooling2D`, then `Flatten`, a 64-unit dense layer, and a softmax head. The VGG16 run freezes the ImageNet backbone and trains only the new head.

| Model | Convolution filters | Head | Parameters | Checkpoint |
| --- | --- | --- | ---: | --- |
| Baseline CNN | 32, 32, 32 | Dense 64, softmax 5 | 838,981 | `Palm_Model.keras` |
| Deeper CNN | 32, 32, 64, 128, 256 | Dense 64, softmax 5 | 545,509 | `aPalm_Model.keras` |
| Wide CNN | 128, 256, 512, 512, 1024 | Dense 64, softmax 4 | 9,148,484 | `a3Palm_Model.keras` |
| Wider mid CNN | 64, 64, 128, 256, 512 | Dense 64, softmax 5 | 1,883,205 | `a4Palm_Model.keras` |
| VGG16 transfer | ImageNet VGG16, frozen | Dense 256, dropout 0.5, softmax 4 | 17,992,772 (3,278,084 trainable) | `VGG16Palm_Model.keras` |

Custom models train for 75 epochs. VGG16 trains for 10. The dataset has four classes; several heads are still declared with five softmax units.

The fit calls under the wide and wider-mid sections train `aModel` (the deeper CNN) again, and those runs write `a3Palm_Model.keras` and `a4Palm_Model.keras`. The 0.90 scores below are from those checkpoints.

## Results

Test scores are from the saved best-validation-loss checkpoint, evaluated on the 20-image test set.

| Checkpoint | Test accuracy | Test loss |
| --- | ---: | ---: |
| `Palm_Model.keras` | 0.850 | 0.276 |
| `aPalm_Model.keras` | 0.80 | 0.361 |
| `a3Palm_Model.keras` | 0.90 | 0.256 |
| `a4Palm_Model.keras` | 0.90 | 0.374 |
| `VGG16Palm_Model.keras` | 1.00 | ~0.00007 |

The baseline model reaches about 0.91 training accuracy and 0.85 validation accuracy by epoch 75. On the 40 validation images its confusion matrix is 0.82 accurate:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| My palm | 0.80 | 0.80 | 0.80 | 10 |
| Person 2 | 0.88 | 0.70 | 0.78 | 10 |
| Person 3 | 0.77 | 1.00 | 0.87 | 10 |
| Person 4 | 0.89 | 0.80 | 0.84 | 10 |

VGG16 reaches 1.00 validation accuracy at epoch 10 and classifies every test image correctly. That test set is only 20 images.

## Run the notebook

Python 3 with Jupyter (or Google Colab). The notebook was executed with TensorFlow Keras.

```bash
pip install tensorflow matplotlib numpy scikit-learn seaborn
jupyter notebook Untitled1.ipynb
```

Put the `Palm` directory described above in the working directory before the data-loading cells. Training writes the `.keras` checkpoints into that same directory.
