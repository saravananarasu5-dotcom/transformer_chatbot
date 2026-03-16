import json
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def main():
    # Load the dataset
    with open('dataset.json', 'r') as file:
        data = json.load(file)

    intents = data['intents']

    # Prepare data structures
    patterns = []
    tags = []
    responses = {}

    # Extract patterns, tags, and responses
    for intent in intents:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
        responses[intent['tag']] = intent['responses']

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Function to clean and lemmatize text
    def clean_text(text):
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    # Clean all patterns
    cleaned_patterns = [clean_text(pattern) for pattern in patterns]

    # Create TF-IDF vectorizer and transform patterns
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_patterns)

    # Save the processed data
    model_data = {
        'vectorizer': vectorizer,
        'X': X,
        'tags': tags,
        'responses': responses
    }

    with open('chatbot_model.pkl', 'wb') as file:
        pickle.dump(model_data, file)

    print("Model training complete")

if __name__ == "__main__":
    main()