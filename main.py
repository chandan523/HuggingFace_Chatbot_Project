import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env
import os
sec_key = os.environ.get("HF_TOKEN")
huggingface_token = os.environ.get("HUGGINGFACE_API_TOKEN")

from langchain_huggingface import HuggingFaceEndpoint

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length = 128, temperature=0.7, token=sec_key)

llm.invoke("What is mechine learning?")

from langchain import PromptTemplate, LLMChain

import datetime
current_date = datetime.date.today()
current_year = current_date.year

question = "who won the world cup in 2011?"
template = """Question: The current year is """ + str(current_year) + """. {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke(question)

# HuggingFace Pipeline

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
hf = HuggingFacePipeline(pipeline=pipe)

hf

hf.invoke("What is AI?")

import re
import long_responses as long
from fuzzywuzzy import fuzz

def message_probability(user_message, recognised_words, single_response=False, required_words=[]):
    message_certainty = 0
    has_required_words = True

    # Counts how many words are present in each predefined message
    for word in user_message:
        if word in recognised_words:
            message_certainty += 1

    # Calculates the percent of recognised words in a user message
    percentage = float(message_certainty) / float(len(recognised_words))

    for word in required_words:
        if word not in user_message:
            has_required_words = False
            break

    if has_required_words or single_response:
        return int(percentage * 100)
    else:
        return 0
    
from fuzzywuzzy import fuzz

def check_all_messages(message):
    highest_prob_list = {}

    def response(bot_response, list_of_words, single_response=False, required_words=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, single_response, required_words)

    # Predefined responses and keywords
    responses = {
        "Hello!": ['hello', 'hi', 'hey', 'greetings'],
        "I am doing well, thank you!": ['how', 'are', 'you'],
        "You're welcome!": ['thank', 'thanks'],
    }

    # Calculate similarity using fuzzywuzzy
    for bot_response, keywords in responses.items():
        similarity = fuzz.token_set_ratio(message, " ".join(keywords))  # Use token_set_ratio for better results
        if similarity > 50:  # Adjust threshold as needed
            highest_prob_list[bot_response] = similarity


    # Return the best match and its probability
    if highest_prob_list:  # Check if there's a match
        best_match = max(highest_prob_list, key=highest_prob_list.get)
        return best_match, highest_prob_list[best_match]
    else:
        return None, 0  # No match found

def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    # Get the best match and probability
    best_match, probability = check_all_messages(split_message)

    # Check if the probability is below a threshold (e.g., 50)
    if probability < 10:  # Adjust threshold as needed
        return long.unknown()
    else:
        return best_match


while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Chatbot: Goodbye! Have a great day!")
        break

    # Get the response and probability
    response, probability = check_all_messages(re.split(r'\s+|[,;?!.-]\s*', user_input.lower()))

    # Use LLM if probability is below the threshold
    if probability < 10:
        response = llm.invoke(user_input)
    else:
        response = get_response(user_input)  # Use pre-defined response

    print("Chatbot:", response)