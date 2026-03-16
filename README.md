# Autonomous Chatbot using Transformer Model (Python & AI/ML)

## Project Description

This is a beginner-friendly mini project that demonstrates how to build an autonomous chatbot using Python and AI/ML techniques. The chatbot uses a dataset of predefined intents and patterns, and falls back to GPT-2 (a transformer model) for generating responses when no predefined match is found.

The project includes:
- A JSON dataset with chatbot intents
- A training script to process the data
- A console-based chatbot application
- Integration with HuggingFace Transformers for AI-powered responses

## Installation Steps

1. Ensure you have Python 3.10 or higher installed on your system.

2. Clone or download this project to your local machine.

3. Open the project folder in Visual Studio Code.

4. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

## How to Run the Training Script

The training script processes the dataset and creates a model file that the chatbot will use.

1. Open a terminal in VS Code (Ctrl + `).

2. Run the training script:
   ```
   python train_model.py
   ```

3. You should see "Model training complete" when finished.

## How to Start the Chatbot

1. After training the model, run the chatbot:
   ```
   python chatbot.py
   ```

2. The chatbot will start and display:
   ```
   Autonomous Transformer Chatbot
   Type 'exit' to stop
   You:
   ```

3. Type your messages and press Enter to chat with the bot.

4. Type 'exit' to stop the conversation.

## Example Chatbot Conversation

```
Autonomous Transformer Chatbot
Type 'exit' to stop

You: hello
Bot: Hello! How can I help you today?

You: what is python
Bot: Python is a high-level programming language known for its simplicity and readability.

You: tell me about machine learning
Bot: Machine learning is a subset of AI that allows systems to learn from data.

You: goodbye
Bot: Goodbye! Have a great day!

You: exit
Bot: Goodbye!
```

## Project Structure

- `dataset.json`: Contains the chatbot intents with patterns and responses
- `train_model.py`: Script to process the dataset and train the model
- `chatbot.py`: Main chatbot application
- `requirements.txt`: List of required Python packages
- `README.md`: This documentation file

## Technologies Used

- Python 3.10+
- NLTK (Natural Language Toolkit) for text processing
- Scikit-learn for TF-IDF vectorization and similarity matching
- HuggingFace Transformers for GPT-2 integration
- PyTorch (automatically installed with transformers)

## Learning Outcomes

By working with this project, you'll learn:
- How to structure a Python AI/ML project
- Text preprocessing and lemmatization
- Vectorization techniques (TF-IDF)
- Similarity matching for intent classification
- Integration with pre-trained transformer models
- Building interactive console applications

## Customization

You can customize the chatbot by:
- Adding more intents to `dataset.json`
- Adjusting the similarity threshold in `chatbot.py`
- Modifying the GPT-2 generation parameters
- Adding more advanced NLP techniques

Happy coding!