import pickle
import random
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def main():
    # Load the trained model data
    with open('chatbot_model.pkl', 'rb') as file:
        model_data = pickle.load(file)

    vectorizer = model_data['vectorizer']
    X = model_data['X']
    tags = model_data['tags']
    responses = model_data['responses']

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Function to clean and lemmatize text
    def clean_text(text):
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    # Load GPT-2 model for fallback response generation
    generator = pipeline('text-generation', model='gpt2')

    print("Autonomous Transformer Chatbot")
    print("Type 'exit' to stop")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break

        # Clean the user input
        cleaned_input = clean_text(user_input)

        # Vectorize the input
        input_vector = vectorizer.transform([cleaned_input])

        # Compute cosine similarities
        similarities = cosine_similarity(input_vector, X).flatten()

        # Find the best matching pattern
        best_index = similarities.argmax()
        best_similarity = similarities[best_index]

        # Threshold for considering a match
        threshold = 0.3

        if best_similarity > threshold:
            # Get the tag and a random response
            tag = tags[best_index]
            response = random.choice(responses[tag])
            print(f"Bot: {response}")
        else:
            # Generate response using GPT-2
            prompt = f"User: {user_input}\nBot:"
            generated = generator(prompt, max_length=50, num_return_sequences=1, pad_token_id=50256)[0]['generated_text']
            # Extract the bot response
            if '\nBot:' in generated:
                bot_response = generated.split('\nBot:')[1].strip()
            else:
                bot_response = generated.replace(prompt, '').strip()
            print(f"Bot: {bot_response}")

if __name__ == "__main__":
    main()