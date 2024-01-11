import json
import random
import requests
import bs4
import datetime
import torch
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd  # Add this import for working with CSV files

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "CEB Assistance"

# Define the path to the log file
log_file = "unanswered_questions.log"

# Cache to store answers
answer_cache = {}

# Function to split response into text and link parts
def split_response(response):
    # Check if the response contains a link (e.g., "https://www.example.com")
    if "https://" in response:
        parts = response.split("https://")
        if len(parts) > 1:
            text_part = parts[0]
            link_part = "https://" + parts[1]
        else:
            text_part = response
            link_part = None
    else:
        text_part = response
        link_part = None
    return text_part, link_part

# Function to log unanswered questions
def log_unanswered_question(question):
    with open(log_file, "a") as log:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"{timestamp} - Unanswered Question: {question}\n")

# Function to find the answer for a given question
def find_answer(user_question):
    user_question = user_question.lower()  # Convert user's question to lowercase

    # Check if the answer is in the cache
    if user_question in answer_cache:
        return answer_cache[user_question]

    # If no answer is found, return a default message
    return None

# Function to get a response based on user input
def get_response(msg):

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.95:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                # Split the response into text and link parts
                text_part, link_part = split_response(response)

                if link_part:
                    # Format the link part as HTML
                    response = f'{text_part}<a href="{link_part}" target="_blank"><button class="btn">Click Here</button></a>'

                return response
            
    answer = find_answer(msg)
    if answer:
        # Split the answer into text and link parts
        text_part, link_part = split_response(answer)

        if link_part:
            # Format the link part as HTML
            answer = f'{text_part}<a href="{link_part}"></a>.'

        return answer
    
    # If no response was found in intents or cache, log the unanswered question
    log_unanswered_question(msg)
    
    return "I couldn't find an answer to your question. Please contact the hotline via 1987."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        question = input("User : ")
        if question.lower() == "quit":
            break
        response = get_response(question)
        print("Bot : ", response)
