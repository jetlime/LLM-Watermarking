import tiktoken
import sys

from os import getenv
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Tuple define cost of input and output per token, in U.S $ 
price_per_token = {"davinci-002": (0.000002, 0.000002), "gpt-3.5-turbo-1106": (0.000001,0.000001),  "babbage-002": (0.0000004, 0.0000004), "gpt-4o": (0.000005, 0.000015), "gpt-4-turbo": (0.00001, 0.00003)} 

# Gather the number of tokens and char on large text sample to compute the average rate of tokens per char
with open("./cost-analysis/shakespeare.txt", "r") as file:
    lines = file.readlines()
    raw_text = "".join(lines)

encoder = tiktoken.encoding_for_model("text-davinci-002")
raw_tokens = encoder.encode(raw_text)
avg_token_per_char =  len(raw_tokens) / len(raw_text)



def cost(characters_amount, model):
    global avg_token_per_char
    total_tokens = avg_token_per_char * characters_amount
    total_output_tokens = total_tokens
    total_input_tokens = (total_tokens*(total_tokens+1))/2
    return (total_input_tokens * price_per_token[model][0]) + (total_output_tokens * price_per_token[model][1])

# make data
x = np.linspace(0, 100000, 100)

# plot
fig, ax = plt.subplots()

for model, price in price_per_token.items():
    y = cost(x, model)
    ax.plot(x, y, label=model, linewidth=2.0)

avg_characters_academic_paper = 45000
ax.axvline(avg_characters_academic_paper, color='red', linestyle='--', label='Avg Academic Paper')

plt.xlabel("Number of Characters")
plt.ylabel("Cost in U.S $")
plt.title("Cost Analysis of using an infinite Sequence Window")
plt.legend()

plt.savefig("./cost-analysis/cost_per_char_inf_sequence_window.png")



def cost_fixed_sequence_window(characters_amount, model, window_size):
    global avg_token_per_char
    total_tokens = avg_token_per_char * characters_amount
    total_output_tokens = total_tokens
    total_input_tokens = (total_tokens/window_size) * ((window_size*(window_size+1))/2)
    return (total_input_tokens * price_per_token[model][0]) + (total_output_tokens * price_per_token[model][1])

# plot
fig, ax = plt.subplots()

for model, price in price_per_token.items():
    y = cost_fixed_sequence_window(x, model, 10)
    ax.plot(x, y, label=model, linewidth=2.0)

avg_characters_academic_paper = 45000
ax.axvline(avg_characters_academic_paper, color='red', linestyle='--', label='Avg Academic Paper')

plt.xlabel("Number of Characters")
plt.ylabel("Cost in U.S $")
plt.title("Cost Analysis of using a fixed Sequence Window of 10 tokens")
plt.legend()

plt.savefig("./cost-analysis/cost_per_char_fixed_sequence_window_10.png")

def cost_fixed_sequence_window(characters_amount, model, window_size):
    global avg_token_per_char
    total_tokens = avg_token_per_char * characters_amount
    total_output_tokens = total_tokens
    total_input_tokens = (total_tokens/window_size) * ((window_size*(window_size+1))/2)
    return (total_input_tokens * price_per_token[model][0]) + (total_output_tokens * price_per_token[model][1])

# plot
fig, ax = plt.subplots()

for model, price in price_per_token.items():
    y = cost_fixed_sequence_window(x, model, 100)
    ax.plot(x, y, label=model, linewidth=2.0)

avg_characters_academic_paper = 45000
ax.axvline(avg_characters_academic_paper, color='red', linestyle='--', label='Avg Academic Paper')

plt.xlabel("Number of Characters")
plt.ylabel("Cost in U.S $")
plt.title("Cost Analysis of using a fixed Sequence Window of 100 tokens")
plt.legend()

plt.savefig("./cost-analysis/cost_per_char_fixed_sequence_window_100.png")