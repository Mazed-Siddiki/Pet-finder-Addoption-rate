# Pet Adoption Speed Prediction

Predicting how quickly a shelter animal will be adopted, from its listing. The listing has two very different kinds of information, a table of attributes and a photograph, so the project builds a feature representation for each and combines them.

Coursework project, MSc Artificial Intelligence and Robotics.

## The problem

`AdoptionSpeed` is a multi class target describing how long an animal took to be adopted. The tabular part of each listing holds age, breed, colour, fee, health and similar fields. Each listing also has photographs, which the table cannot represent.

## Turning images into features

Photographs were converted into a bag of visual words:

1. **SIFT descriptors** are extracted from every image with OpenCV, giving a variable number of local keypoint descriptors per photograph.
2. **MiniBatchKMeans** clusters all of those descriptors into a fixed vocabulary of visual words.
3. Each image is then described by **how often each visual word appears in it**, which turns a variable length set of descriptors into a fixed length vector that can sit next to the tabular columns.

This is what makes the two sources combinable: the image ends up as just more columns.

## Models

Five classifiers were trained on the combined features, each tuned with `GridSearchCV`:

| Model | Best score |
|---|---|
| Gradient Boosting | 0.30 |
| Random Forest | 0.2834 |
| SVM | 0.276 |
| XGBoost | 0.2511 |
| Logistic Regression | 0.2455 |

The predictions of the individual models were then collected into a new dataset and combined, rather than relying on a single classifier.

## Honest note on the numbers

These scores are low in absolute terms. Adoption speed is genuinely hard to predict from a listing, and the classes are imbalanced, so the interesting part of this project is the multimodal pipeline and the comparison between models, not the headline accuracy.

## Files

```
Pet_Adoption_Speed.ipynb   full pipeline, 110 code cells
train.csv, test.csv        tabular listings
test_images.zip            photographs
results.csv                predictions
```

## Running it

```
pip install numpy pandas scikit-learn xgboost opencv-python scikit-image matplotlib seaborn tqdm
jupyter notebook Pet_Adoption_Speed.ipynb
```
