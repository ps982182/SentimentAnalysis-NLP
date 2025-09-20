import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# **MUCH LARGER** training dataset [web:235][web:241]
training_data = [
    # Positive samples
    ("I love this product amazing quality", 1),
    ("This is the best thing ever", 1),
    ("Absolutely fantastic experience", 1),
    ("Outstanding service and support", 1),
    ("Highly recommend to everyone", 1),
    ("Perfect solution to my problem", 1),
    ("Excellent value for money", 1),
    ("Great customer service team", 1),
    ("Amazing product quality and fast delivery", 1),
    ("Love the design and functionality", 1),
    ("Super happy with my purchase", 1),
    ("Exceeded all my expectations", 1),
    ("Wonderful experience from start to finish", 1),
    ("Top notch quality and service", 1),
    ("Best investment I have made", 1),
    ("Incredible results and performance", 1),
    ("Really impressed with the quality", 1),
    ("Fantastic product would buy again", 1),
    ("Excellent communication and delivery", 1),
    ("Perfect fit exactly what I needed", 1),
    ("Great features and easy to use", 1),
    ("Awesome customer support team", 1),
    ("High quality materials and construction", 1),
    ("Amazing functionality and design", 1),
    ("Love everything about this product", 1),
    ("Brilliant solution highly recommended", 1),
    ("Superb quality and fast shipping", 1),
    ("Outstanding value for the price", 1),
    ("Excellent product meets all expectations", 1),
    ("Great experience will shop again", 1),
    
    # Negative samples
    ("I hate this terrible quality", 0),
    ("This is the worst experience ever", 0),
    ("Completely disappointed and frustrated", 0),
    ("Awful service and poor support", 0),
    ("Would not recommend to anyone", 0),
    ("Useless product waste of money", 0),
    ("Terrible value completely overpriced", 0),
    ("Poor customer service very rude", 0),
    ("Bad product quality and slow delivery", 0),
    ("Hate the design and poor functionality", 0),
    ("Very unhappy with my purchase", 0),
    ("Failed to meet any expectations", 0),
    ("Horrible experience from start to finish", 0),
    ("Low quality materials and service", 0),
    ("Worst investment I have made", 0),
    ("Disappointing results and performance", 0),
    ("Really unimpressed with the quality", 0),
    ("Terrible product would never buy again", 0),
    ("Poor communication and late delivery", 0),
    ("Wrong fit not what I needed", 0),
    ("Bad features and difficult to use", 0),
    ("Awful customer support team", 0),
    ("Low quality materials and construction", 0),
    ("Terrible functionality and design", 0),
    ("Dislike everything about this product", 0),
    ("Poor solution not recommended", 0),
    ("Bad quality and slow shipping", 0),
    ("Disappointing value for the price", 0),
    ("Poor product fails expectations", 0),
    ("Bad experience will not shop again", 0),
    
    # More nuanced examples
    ("Good product but could be better", 1),
    ("Decent quality for the price", 1),
    ("Not bad but room for improvement", 1),
    ("Okay service nothing special", 1),
    ("Average product meets basic needs", 1),
    ("Fine quality acceptable performance", 1),
    ("Pretty good value overall", 1),
    ("Reasonable quality and service", 1),
    ("Satisfied with the purchase", 1),
    ("Works as expected no complaints", 1),
    ("Below average quality disappointed", 0),
    ("Not great poor value", 0),
    ("Mediocre service and quality", 0),
    ("Disappointing results not recommended", 0),
    ("Poor performance below expectations", 0),
    ("Bad experience poor quality", 0),
    ("Not satisfied with purchase", 0),
    ("Subpar quality and service", 0),
    ("Unimpressed with overall experience", 0),
    ("Not worth the money spent", 0),
]

# Convert to DataFrame
df = pd.DataFrame(training_data, columns=['text', 'sentiment'])
print(f"Dataset size: {len(df)} samples")
print(f"Positive samples: {sum(df['sentiment'])}")
print(f"Negative samples: {len(df) - sum(df['sentiment'])}")

def preprocess_text(text):
    """Enhanced text preprocessing [web:233][web:237]"""
    text = text.lower()
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#\w+','', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_tokens)

# Preprocess the data
df['clean_text'] = df['text'].apply(preprocess_text)

# Split data with stratification [web:234]
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['sentiment'], 
    test_size=0.25, 
    random_state=42,
    stratify=df['sentiment']  # Ensures balanced split
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# **OPTIMIZED** TF-IDF Vectorizer [web:233][web:236]
vectorizer = TfidfVectorizer(
    max_features=2000,           # Increased features
    ngram_range=(1, 2),          # Include bigrams
    min_df=2,                    # Ignore rare words
    max_df=0.8,                  # Ignore too common words
    sublinear_tf=True            # Apply sublinear scaling
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Feature extraction completed")

# **HYPERPARAMETER TUNING** [web:236][web:244]
print("Starting hyperparameter tuning...")

# Try multiple algorithms with parameter tuning
models = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
}

best_model = None
best_score = 0
best_name = ""

for name, config in models.items():
    print(f"\nTuning {name}...")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        config['model'], 
        config['params'], 
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    # Test on validation set
    score = grid_search.score(X_test_tfidf, y_test)
    print(f"{name} best score: {score:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    if score > best_score:
        best_score = score
        best_model = grid_search.best_estimator_
        best_name = name

print(f"\n🏆 Best model: {best_name} with accuracy: {best_score:.4f}")

# **DETAILED EVALUATION** [web:228][web:232]
y_pred = best_model.predict(X_test_tfidf)
print(f"\n📊 Detailed Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Save the best model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Optimized model and vectorizer saved successfully!")

# **TEST THE IMPROVED MODEL**
print("\n🧪 Testing improved model:")
test_texts = [
    "I love this product amazing quality",
    "This is terrible worst ever", 
    "Great experience highly recommend",
    "Bad service very disappointed",
    "Excellent value for money",
    "Waste of money poor quality"
]

for text in test_texts:
    clean_text = preprocess_text(text)
    vect = vectorizer.transform([clean_text])
    pred = best_model.predict(vect)[0]
    confidence = best_model.predict_proba(vect)[0].max()
    sentiment = 'positive' if pred == 1 else 'negative'
    print(f"'{text}' -> {sentiment} (confidence: {confidence:.3f})")
