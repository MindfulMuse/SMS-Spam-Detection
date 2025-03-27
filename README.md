# SMS Spam Detection

## Overview
SMS Spam Detection is a machine learning model that takes an SMS as input and predicts whether the message is spam or not spam. The model integrates **language detection, translation, text preprocessing, feature extraction, and classification** to improve accuracy. It is built using Python and deployed on the web using **Streamlit**.

### Features:
- **Language Detection:** Identifies the language of the SMS.
- **Translation:** Converts non-English messages to English for better analysis.
- **Text Preprocessing:** Cleans and processes text by removing stopwords, punctuation, and performing stemming.
- **Feature Extraction:** Uses **TF-IDF vectorization** to convert text into numerical features.
- **Spam Classification:** Implements machine learning models like **Naïve Bayes, Random Forest, and Extra Trees** to classify messages.
- **Deployment:** Accessible through a user-friendly web interface built with **Streamlit**.

## Technology Used

| Library      | Purpose                                               |
|--------------|-------------------------------------------------------|
| numpy        | Numerical operations                                  |
| pandas       | Data handling & processing                            |
| scikit-learn | Machine Learning models                               |
| nltk         | Natural Language Processing (Stopwords, Tokenization) |
| joblib       | Model saving & loading                                |
| streamlit    | Web application framework                             |
| googletrans  | Automatic language translation                        |

### File Structure 

```
sms-spam-detection/
│── model.pkl          # Trained machine learning model
│── vectorizer.pkl     # TF-IDF Vectorizer
│── app.py             # Main Streamlit app
│── sms-spam.ipynb     # Text preprocessing functions
│── spam.csv           # SMS spam dataset
└── README.md          # Project Documentation
```

### Data Collection
The SMS Spam Collection dataset was collected from Kaggle, which contains over 5,500 SMS messages labeled as either spam or not spam.
You can access the dataset from [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Installations

Install dependencies :

```
pip install numpy pandas scikit-learn nltk joblib streamlit googletrans==4.0.0-rc1
```

### 1. Text Preprocessing
- **Lowercasing:** Converted all text to lowercase.
- **Tokenization:** Converted text into individual words.
- **Stopwords & Punctuation Removal:** Removed common words that do not add meaning.
- **Stemming:** Reduced words to their base form using **PorterStemmer**.
- 
### 2. Data Cleaning and Preprocessing
The data was cleaned by handling null and duplicate values, and the "type" column was label-encoded. The data was then preprocessed by converting the text into tokens, removing special characters, stop words and punctuation, and stemming the data. The data was also converted to lowercase before preprocessing.

### 3. Exploratory Data Analysis
Exploratory Data Analysis (EDA) was conducted to extract meaningful insights from the dataset. Key text-based features such as **character count, word count, and sentence count** were computed for each message. The relationships between these variables were analyzed through correlation metrics. Various **visualization techniques** were employed, including **bar charts, pie charts, box plots (five-number summaries), and heatmaps** to understand data distribution and patterns. Additionally, **word clouds** were generated to highlight frequently occurring words in both **spam and non-spam messages**, with a separate visualization showcasing the most common words found in spam texts.

### 4. Model Building and Selection
Multiple classifier models were tried, including NaiveBayes, random forest, KNN, decision tree, logistic regression, ExtraTreesClassifier, and SVC. The best classifier was chosen based on precision, with a precision of 100% achieved.

## 5. Model Deployment
To deploy the model, the following **two files** were saved using `pickle`:
- **vectorizer.pkl** → Stores the trained TF-IDF vectorizer.
- **model.pkl** → Stores the trained model.

## Application (`app.py`) 

The trained model was deployed using **Streamlit** with a simple web-based UI.  

## **Key Functionalities**  

### **Preprocessing**
- Converts text to **lowercase**.  
- Tokenizes words and removes **special characters**.  
- Eliminates **stopwords** and applies **stemming** using **PorterStemmer**.  

### **Language Detection & Translation**
- Detects input message **language** using **Google Translate API**.  
- If not in **English**, the text is **automatically translated**.  

### **Feature Extraction & Prediction**
- Uses **TF-IDF vectorizer** to transform text into **numerical features**.  
- The **pre-trained ML model** predicts whether a message is **Spam or Not Spam**.  

### **User-Friendly Web Interface**
- Built with **Streamlit**, allowing users to enter an **SMS message**.  
- Displays the **detected language, translated text**, and **classification result**.  
- Provides a **🚨 Spam warning** or **✅ Safe message confirmation**.  

## **Web Deployment**
The model was **deployed online** using **Streamlit**, where users can enter a message and receive an instant prediction.  

---

## Usage
To use the SMS Spam Detection model on your own machine, follow these steps:

+ Run the model using 
```
streamlit run app.py.
```
+ Visit localhost:8501 on your web browser to access the web app.


## Contributions
Contributions to this project are welcome. If you find any issues or have any suggestions for improvement, please open an issue or a pull request on this repository.
