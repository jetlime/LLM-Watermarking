import matplotlib.pyplot as plt
import argparse


price_per_token = {
    "davinci-002": (0.000002, 0.000002),
    "gpt-3.5-turbo-1106": (0.000001, 0.000001),
    "babbage-002": (0.0000004, 0.0000004),
    "gpt-4o": (0.000005, 0.000015),
    "gpt-4-turbo": (0.00001, 0.00003),
}

# Prompt older foundational models, which have been
# trained to predict next sequence of tokens
def send_prompt_completion(client, config, input_prompt):
    response = client.completions.create(
        model=config.model_id,
        prompt=input_prompt,
        max_tokens=1,
        temperature=0.7,
        seed=1234,
        logprobs=config.frequency_bins,
    )
    logprobs = response.choices[0].logprobs.top_logprobs
    return logprobs[0]


# Prompt newer models, finetuned on chat instruction
def send_prompt_chat_completion(client, config, input_prompt):
    response = client.chat.completions.create(
        model=config.model_id,
        messages=[
            {
                "role": "system",
                "content": "Complete the following sentence using a single word.",
            },
            {"role": "user", "content": input_prompt},
        ],
        seed=1234,
        n=1,
        logprobs=True,
        top_logprobs=20,
    )
    next_word_log_probs = {}
    for logprob in response.choices[0].logprobs.content[0].top_logprobs:
        next_word_log_probs[logprob.token] = logprob.logprob
    return next_word_log_probs


def normalise_list(data):
    return [x / sum(data) for x in data]


def prompt_template(sequence):
    return f"Complete the sentence, provide only a single word as output: {sequence}"


def plot_results(word_selection_distribution, input_tokens, output_tokens, config, raw):
    x_labels = list(range(len(word_selection_distribution)))

    # Plotting the histogram
    plt.bar(x_labels, normalise_list(word_selection_distribution), tick_label=x_labels)

    # Adding title and labels
    plt.xlabel(
        "Index in the probability distribution of the LLM, of the word chosen by the Human"
    )
    plt.ylabel("Distribution over 1")
    request_cost = (input_tokens * price_per_token[config.model_id][0]) + (
        output_tokens * price_per_token[config.model_id][1]
    )
    plt.title(
        f"Total API cost of U.S ${round(request_cost, 3)} ({output_tokens} API calls) for {len(raw)} characters"
    )

    # Display the plot
    output_filename = config.input_file_name.split("/")[-1].split(".")[0].split("_")
    plt.savefig(
        f"./results_{config.model_id}/plot_{output_filename[0]}_{output_filename[1]}.png"
    )

    # Save distribution to file for further analysis
    with open(
        f"./results_{config.model_id}/result_{output_filename[0]}_{output_filename[1]}",
        "w",
    ) as f:
        for s in word_selection_distribution:
            f.write(str(s) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Watermarking LLMs.")
    parser.add_argument(
        "input_file_name", type=str, help="Path to the input text file."
    )
    parser.add_argument(
        "model_id",
        metavar="open_ai_model_id",
        type=str,
        choices=["davinci-002", "babbage-002", "gpt-4o", "gpt-3.5-turbo-1106"],
        help="The ID of the judge LLM, as specified by the OPEN AI API documentation: https://platform.openai.com/docs/models. Supported models are 'davinci-002', 'babbage-002' 'gpt-4o' and 'gpt-3.5-turbo-1106'.",
    )
    parser.add_argument(
        "context_window",
        metavar="context_window",
        type=int,
        action="store",
        default=10,
        help="The context window size (an integer). Defaults to 10.",
        nargs="?",
    )

    parser.add_argument(
        "frequency_bins",
        metavar="frequency_bins",
        type=int,
        action="store",
        default=20,
        help="The number of bins to consider in the frequency distribution. Defaults to 20.",
        nargs="?",
    )
    return parser.parse_args()
