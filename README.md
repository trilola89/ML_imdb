# ML_imdb
Sentiment analysis of imdb movie reviews using TensorFlow

### Exploratory Data Analysis (`eda.ipynb`)

The `eda.ipynb` notebook performs a comprehensive analysis and preprocessing of the IMDB movie reviews dataset. The notebook is organized into the following key stages:

#### 1. Data Access
   - **Download:** Utilizes the Kaggle API to download the IMDB dataset.
   - **Load:** Reads the dataset into a Pandas DataFrame for further analysis and preprocessing.

#### 2. Preprocessing and Feature Engineering
   - **Handling Null Values:** Checks for missing values and handles them if necessary.
   - **Review Length Analysis:** Computes the length of each review and adds it as a new feature.
   - **Text Preprocessing:** Tokenizes and preprocesses the reviews, converting them into a format suitable for machine learning models.
   - **Label Encoding:** Transforms the sentiment labels into numerical values using one-hot encoding.
   - **Data Type Optimization:** Converts specific columns to more efficient data types (e.g., `int16`) to optimize memory usage.

#### 3. Exploratory Data Analysis
   - **Sentiment Distribution:** Visualizes the distribution of positive and negative sentiments in the dataset.
   - **Review Length Histogram:** Plots a histogram of review lengths to understand the distribution of review sizes.
   - **Word Frequency Analysis:** Analyzes the most common words in the dataset using a word frequency counter.

#### 4. Data Saving
   - **HDF5 File:** Saves the processed and transformed DataFrame into an HDF5 file, facilitating easy access for future modeling tasks.

The `eda.ipynb` notebook serves as a foundational step in the project, ensuring that the data is cleaned, preprocessed, and analyzed to facilitate subsequent machine learning tasks. By applying robust preprocessing techniques and conducting insightful exploratory analysis, it lays the groundwork for building and evaluating predictive models.

### IMDB Movie Review Classification (`imdb_model.ipynb`)

The `imdb_model.ipynb` notebook demonstrates the complete process of building and evaluating machine learning models for sentiment analysis on the IMDB movie reviews dataset. The notebook is organized into the following key stages:

#### 1. Data Preprocessing
   - **Text Cleaning:** Utilizes custom functions to remove HTML tags, punctuation, and stopwords, and applies tokenization.
   - **Padding:** Ensures that all sequences have the same length by padding shorter sequences with zeros.
   - **Label Encoding:** Encodes the sentiment labels into numerical values.

#### 2. Model Building
   - **Logistic Regression Model:** Implements a baseline logistic regression model to predict the sentiment of the movie reviews.
   - **Deep Learning Model:** Constructs a deep learning model using an Embedding layer followed by two LSTM layers and a Dense output layer with softmax activation.

#### 3. Model Training
   - **Training Configuration:** Specifies hyperparameters such as epochs, batch size, and optimizer.
   - **Training Execution:** Trains the deep learning model on the training dataset and evaluates it on the validation dataset.

#### 4. Model Evaluation
   - **Accuracy and F1 Score:** Evaluates the logistic regression model using accuracy and F1 score.
   - **Loss and Accuracy:** Evaluates the deep learning model using loss and accuracy metrics.

#### 5. Model Saving
   - **Saved Model:** Saves the trained deep learning model to an HDF5 file for future deployment and use.

The `imdb_model.ipynb` notebook provides an end-to-end solution for sentiment analysis on movie reviews, starting from preprocessing to modeling, training, and evaluation. It showcases both traditional machine learning
