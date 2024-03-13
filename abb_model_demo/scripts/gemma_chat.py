"""
Script to chat with a Gemma instruction-tuned LLM model.

I used this script to generate docstrings for the functions in this script!


See
Gemma https://github.com/google/gemma_pytorch
Gemma PyTorch docs https://ai.google.dev/gemma/docs/pytorch_gemma
Gemma formatting docs for instruction tuned models
 https://ai.google.dev/gemma/docs/formatting

Weights D/L from https://www.kaggle.com/models/google/gemma/frameworks/pyTorch
"""

from gemma.config import GemmaConfig, get_config_for_7b, get_config_for_2b
from gemma.model import GemmaForCausalLM
import contextlib
import os
import torch
import argparse

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
#MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)


def load_model(model_variant, device_str, weights_base_dir):
    """Loads a model from a given variant and weight directory.

    Args:
        model_variant (str): The name of the model variant to load.
        device_str (str): The device to load the model on.
        weights_base_dir (str): The base directory for the weights file.

    Returns:
        tuple(model, device)

        torch.nn.Module: The loaded model.
        torch.device: The device to use for model calculations.
    """

    weights_dir = os.path.join(weights_base_dir,
                               f"gemma-{model_variant}")
    model_config = get_config_for_2b() if "2b" in model_variant \
        else get_config_for_7b()
    model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")

    device = torch.device(device_str)
    print(f"DEVICE: {device}")
    print("====== loading model ======")
    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config)
        ckpt_path = os.path.join(weights_dir, f'gemma-{model_variant}.ckpt')
        print(f"load weights from [{ckpt_path}]")
        model.load_weights(ckpt_path)
        model = model.to(device).eval()
    return model, device


def templated_prompt(user_query):
    """
    This function takes a user query as input and produces a templated response.

    Args:
        user_query (str): The user's query.

    Returns:
        str: The templated response.
    """

    prompt = (
        USER_CHAT_TEMPLATE.format(
            prompt=user_query
        )
        + "<start_of_turn>model\n"
    )
    return prompt


def get_gen_response(prompt, model, device, output_len=300):
    """
    Generates a response from a model.

    Args:
        prompt (str): The prompt to generate a response for.
        model (torch.nn.Module): The model to generate the response from.
        device (str): The device to use for generation.
        output_len (int, optional): The length of the response to generate.

    Returns:
        str: The generated response.
    """
    resp = model.generate(
        prompt,
        device=device,
        output_len=output_len
    )
    return resp


def do_uquery_loop(model, device, output_len=300):
    """
    Run a loop that prompts the user for input queries and then
    generates responses from a conversational AI model. The full
    conversation is tracked and passed back to the model with
    each subsequent query. The Gemma instruction-tuned conversation
    syntax is applied.

    TODO: Automatically drop old conversation pieces out to prevent
    the conversation buffer getting too big. Base the drop-out on
    the expected context-size of the model being used.

    Args:
        model (torch.nn.Module): The model to generate the response from.
        device (str): The device to use for generation.
        output_len (int, optional): The length of the response to generate.
          Defaults to 300.

    Returns:
        None
    """

    entire_chat = ""
    # begin query loop
    while True:
        # allow multi-line inputs
        uline = input("< Enter query below. Empty line stops input. DONE quits. >\n")
        # build up multi-line input
        uq = uline
        while len(uline)>0:
            uline = input()
            uq = "\n".join([uq, uline])
        if uq.strip()=="DONE":
            break
        print("OK, generating response...")
        # format this (latest) query
        last_uchat = templated_prompt(user_query=uq)
        # update the full chat
        entire_chat += last_uchat
        res = get_gen_response(prompt=entire_chat,
                               model=model,
                               device=device,
                               output_len=output_len
                               )
        entire_chat += f"{res}<end_of_turn>\n"
        print(res)
    print("DONE")


def get_args():
    """
    Parses command line arguments and returns the arguments object

    :return: the parsed command line arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("-m", "--model_variant",
                        default="2b-it",
                        choices=["2b-it", "7b-it-quant"],
                        help="The Gemma model variant to use"
                        )
    parser.add_argument("-d", "--device_str",
                        default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="The device to use for model predictions"
                        )
    parser.add_argument("-w", "--weights_base_dir",
                        default="/mnt/WDB_Data_L/DATA/MODEL_WEIGHTS/",
                        help=("Base directory where Gemma model variant(s) "
                              "are stored. The subdirectories are expected "
                              "to be of the form [gemma-VARIANT].")
                        )
    parser.add_argument("-l", "--max_output_len",
                        default=300,
                        help="Output length of generated responses"
                        )
    return parser.parse_args()


def main(model_variant, device_str, weights_base_dir, max_output_len):
    model, device = load_model(model_variant=model_variant,
                               device_str=device_str,
                               weights_base_dir=weights_base_dir
                               )
    do_uquery_loop(model=model,
                   device=device,
                   output_len=max_output_len
                   )


if __name__=="__main__":
    args = get_args()
    main(model_variant=args.model_variant,
         device_str=args.device_str,
         weights_base_dir=args.weights_base_dir,
         max_output_len=args.max_output_len
         )

