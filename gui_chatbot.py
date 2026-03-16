
import tkinter as tk
from transformers import pipeline

# Load transformer model
chatbot = pipeline("text-generation", model="gpt2")

# Function to send message
def send_message():
    user_input = entry_box.get()

    if user_input.strip() == "":
        return

    chat_window.insert(tk.END, "You: " + user_input + "\n")

    if user_input.lower() == "exit":
        chat_window.insert(tk.END, "Bot: Goodbye!\n")
        root.quit()
        return

    # Generate response
    response = chatbot(user_input, max_length=40, num_return_sequences=1)
    bot_reply = response[0]["generated_text"]

    chat_window.insert(tk.END, "Bot: " + bot_reply + "\n\n")

    entry_box.delete(0, tk.END)


# GUI Window
root = tk.Tk()
root.title("Autonomous Transformer Chatbot")
root.geometry("500x600")

# Chat display
chat_window = tk.Text(root, bd=1, bg="white", width=50, height=25)
chat_window.pack(padx=10, pady=10)

# Input box
entry_box = tk.Entry(root, width=40)
entry_box.pack(padx=10, pady=10)

# Send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

root.mainloop()