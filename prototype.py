import tiktoken
from os import getenv
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from utils import (
    prompt_template,
    plot_results,
    parse_args,
    send_prompt_chat_completion,
    send_prompt_completion,
)


load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")


def main():
    input_tokens = output_tokens = 0

    config = parse_args()

    print(
        f"Processing file '{config.input_file_name}' with model '{config.model_id}' and context window of {config.context_window}."
    )

    # Read and tokenize the Input file
    encoder = tiktoken.encoding_for_model(config.model_id)
    with open(config.input_file_name, "r") as file:
        lines = file.readlines()
        raw = "".join(lines)
    raw_tokens = encoder.encode(raw)

    client = OpenAI(api_key=GPT_API_KEY)

    # We only consider a probability distributions corresponding to the number of
    # specified Frequency Bins, and we include a bin in case the token is not found
    word_selection_distribution = [0] * config.frequency_bins

    # For all tokens in the document
    for i in tqdm(range(1, len(raw_tokens))):
        # Consider the series of tokens preceding the current token,
        # being as long as the specified context window.
        # THe begining of the document will not a shorter sequence of tokens
        start_index = max(0, i - config.context_window)
        previous_sequence = encoder.decode(raw_tokens[start_index:i])

        next_token = raw_tokens[i]
        input_prompt = prompt_template(previous_sequence)
        input_tokens += len(encoder.encode(input_prompt))

        # Prompt changes based on Judge Model type (foundational or chat finetuned)
        if config.model_id in ["davinci-002", "babbage-002"]:
            next_word_log_probs = send_prompt_completion(client, config, input_prompt)
        elif config.model_id in ["gpt-3.5-turbo-1106"]:
            next_word_log_probs = send_prompt_chat_completion(
                client, config, input_prompt
            )

        output_tokens += 1
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

    plot_results(word_selection_distribution, input_tokens, output_tokens, config, raw)


if __name__ == "__main__":
    main()
