import tiktoken
import sys

from os import getenv
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt

FILE_NAME = sys.argv[1]
MODEL_ID = sys.argv[2]
FREQUENCY_BINS = 21
SEQUENCE_WINDOW_SIZE = 10

input_tokens = output_tokens = 0
price_per_token = {
    "davinci-002": (0.000002, 0.000002),
    "gpt-3.5-turbo-1106": (0.000001, 0.000001),
    "babbage-002": (0.0000004, 0.0000004),
    "gpt-4o": (0.000005, 0.000015),
    "gpt-4-turbo": (0.00001, 0.00003),
}
encoder = tiktoken.encoding_for_model(MODEL_ID)


def prompt_template(sequence):
    return f"Complete the sentence, provide only a single word as output: {sequence}"

with open(FILE_NAME, "r") as file:
    lines = file.readlines()
    raw = "".join(lines)

raw_tokens = encoder.encode(raw)

load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")

client = OpenAI(api_key=GPT_API_KEY)


# We only consider a probability distributions of 5 tokens, and we include a bin in case the token is not found
word_selection_distribution = [0] * FREQUENCY_BINS


def iterative_word_predition(raw_tokens, encoder):
    global input_tokens, output_tokens
    for i in tqdm(range(1, len(raw_tokens))):
        if SEQUENCE_WINDOW_SIZE <= len(raw_tokens[:i]):
            previous_sequence = encoder.decode(raw_tokens[i - SEQUENCE_WINDOW_SIZE : i])
        else:
            previous_sequence = encoder.decode(raw_tokens[:i])
        next_token = raw_tokens[i]
        input_prompt = prompt_template(previous_sequence)
        input_tokens += len(encoder.encode(input_prompt))
        if MODEL_ID == "davinci-002":
            response = client.completions.create(
                model=MODEL_ID,
                prompt=input_prompt,
                max_tokens=1,
                temperature=0.7,
                seed=1234,
                logprobs=FREQUENCY_BINS,
            )
            logprobs = response.choices[0].logprobs.top_logprobs
            next_word_log_probs = logprobs[0]
    
        elif MODEL_ID == "gpt-3.5-turbo-1106":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Complete the following sentence using a single word."},
                    {"role": "user", "content": "I am "},
                ],
                seed=1234,
                n=1,
                logprobs=True,
                top_logprobs=20
            )
            next_word_log_probs = {}
            for logprob in response.choices[0].logprobs.content[0].top_logprobs:
                next_word_log_probs[logprob.token] = logprob.logprob

        output_tokens += 1
        # iterate through the ordered list of the most probable next words
        index = 0
        found_token = False
        for next_word_candidate in next_word_log_probs:
            index += 1
            print(encoder.decode([next_token]), next_word_candidate)
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


def normalize_list(data):
    return [x / sum(data) for x in data]


x_labels = list(range(len(word_selection_distribution)))

# Plotting the histogram
plt.bar(x_labels, normalize_list(word_selection_distribution), tick_label=x_labels)

# Adding title and labels
plt.xlabel(
    "Index in the probability distribution of the LLM, of the word chosen by the Human"
)
plt.ylabel("Distribution over 1")
request_cost = (input_tokens * price_per_token[MODEL_ID][0]) + (
    output_tokens * price_per_token[MODEL_ID][1]
)
plt.title(
    f"Total API cost of U.S ${round(request_cost, 3)} ({output_tokens} API calls) for {len(raw)} characters"
)

# Display the plot
plt.savefig(f"./results_{MODEL_ID}/plot_{FILE_NAME.split('/')[-1].split('.')[0].split('_')[0]}_{FILE_NAME.split('/')[-1].split('.')[0].split('_')[1]}.png")

# Save distribution to file for further analysis
with open(f"./results_{MODEL_ID}/result_{FILE_NAME.split('/')[-1].split('.')[0].split('_')[0]}_{FILE_NAME.split('/')[-1].split('.')[0].split('_')[1]}", "w") as f:
    for s in word_selection_distribution:
        f.write(str(s) +"\n")