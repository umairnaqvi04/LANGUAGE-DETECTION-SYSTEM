import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

warnings.filterwarnings('ignore')

class LanguageDetectionModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        print("📂 Loading data...")
        try:
            self.data = pd.read_csv(self.csv_path)
            if self.data.shape[0] == 0:
                raise ValueError("Dataset empty!")
            print(f"✅ Loaded {len(self.data)} samples from {self.data['Language'].nunique()} languages\n")
            return True
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return False
    
    def preprocess_data(self):
        print("🔧 Preprocessing...")
        try:
            self.data = self.data.dropna()
            self.data = self.data[self.data['Text'].str.len() > 0]
            
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
            X = self.vectorizer.fit_transform(self.data['Text'])
            y = self.data['Language']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"✅ Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}, Features: {self.X_train.shape[1]}\n")
            return True
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return False
    
    def train_models(self):
        print("🤖 Training 3 models...")
        try:
            self.models['SVM'] = LinearSVC(random_state=42, max_iter=2000)
            self.models['SVM'].fit(self.X_train, self.y_train)
            
            self.models['Naive Bayes'] = MultinomialNB()
            self.models['Naive Bayes'].fit(self.X_train, self.y_train)
            
            self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            self.models['Random Forest'].fit(self.X_train, self.y_train)
            
            print("✅ All models trained\n")
            return True
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return False
    
    def evaluate_models(self):
        print("📊 Evaluating models...")
        try:
            for model_name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                self.results[model_name] = accuracy
                print(f"  {model_name}: {accuracy:.2%}")
            
            best = max(self.results, key=self.results.get)
            print(f"\n🏆 Best: {best} ({self.results[best]:.2%})\n")
            return True
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return False
    
    def save_models(self, output_dir='models'):
        print(f"💾 Saving to {output_dir}...")
        try:
            Path(output_dir).mkdir(exist_ok=True)
            pickle.dump(self.models['SVM'], open(f'{output_dir}/language_detection_model.pkl', 'wb'))
            pickle.dump(self.vectorizer, open(f'{output_dir}/vectorizer.pkl', 'wb'))
            pickle.dump(self.models, open(f'{output_dir}/all_models.pkl', 'wb'))
            print("✅ Models saved!\n")
            return True
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return False
    
    def run(self, output_dir='models'):
        print("="*50)
        print("🚀 LANGUAGE DETECTION TRAINING")
        print("="*50 + "\n")
        
        if not self.load_data():
            return False
        if not self.preprocess_data():
            return False
        if not self.train_models():
            return False
        if not self.evaluate_models():
            return False
        if not self.save_models(output_dir):
            return False
        
        print("="*50)
        print("✅ TRAINING COMPLETE!")
        print("="*50)
        return True

if __name__ == "__main__":
    model = LanguageDetectionModel('data/Language_Detection.csv')
    model.run()
