import tiktoken
import sys

from os import getenv
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt

input_tokens = output_tokens = 0
price_per_token = 0.000012 # 12$/1M Input & Output Token for davinci-002

encoder = tiktoken.encoding_for_model("text-davinci-002")


def prompt_template(sequence):
    return f"Complete the sentence, provide only a single word as output: {sequence}"


FILE_NAME = sys.argv[1]
TOKEN_TO_CONSIDER = 7

with open(FILE_NAME, "r") as file:
    lines = file.readlines()
    raw = "".join(lines)

raw_tokens = encoder.encode(raw)

load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")

client = OpenAI(api_key=GPT_API_KEY)


# We only consider a probability distributions of 5 tokens, and we include a bin in case the token is not found
word_selection_distribution = [0] * TOKEN_TO_CONSIDER


def iterative_word_predition(raw_tokens, encoder):
    global input_tokens, output_tokens
    for i in tqdm(range(1, len(raw_tokens))):
        previous_sequence = encoder.decode(raw_tokens[:i])
        next_token = raw_tokens[i]
        input_prompt = prompt_template(previous_sequence)
        input_tokens += len(encoder.encode(input_prompt))
        response = client.completions.create(
            model="davinci-002",
            prompt=input_prompt,
            max_tokens=1,
            temperature=0.7,
            logprobs=TOKEN_TO_CONSIDER,
        )
        output_tokens += 1
        logprobs = response.choices[0].logprobs.top_logprobs
        next_word_log_probs = logprobs[0]
        # iterate through the ordered list of the most probable next words
        index = 0
        found_token = False
        for next_word_candidate in next_word_log_probs:
            index += 1
            if [next_token] == encoder.encode(
                next_word_candidate, allowed_special={"<|endoftext|>"}
            ):
                # place in the density bin corresponding to the index it was found in the log probability distribution
                word_selection_distribution[index - 1] += 1
                found_token = True
                break
        if found_token == False:
            # place in the last density bin
            word_selection_distribution[-1] += 1


iterative_word_predition(raw_tokens, encoder)
print(word_selection_distribution)


x_labels = list(range(len(word_selection_distribution)))

# Plotting the histogram
plt.bar(x_labels, word_selection_distribution, tick_label=x_labels)

# Adding title and labels
plt.xlabel("Index in the probability distribution of the LLM, of the word chosen by the Human")
plt.ylabel("Occurance (#)")
request_cost = (input_tokens + output_tokens) * price_per_token
plt.title(
    f"Total API cost U.S ${round(request_cost, 3)} for a document of {len(raw)} characters"
)

# Display the plot
plt.savefig(f"./results/result_plot-{FILE_NAME.split('/')[-1].split('.')[0]}.png")
