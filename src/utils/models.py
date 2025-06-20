"""Model utilities."""

import gc
import os
from typing import Any

import blobfile as bf
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from utils import config, sae

set_seed(42)

token = os.environ["HUGGING_FACE_HUB_TOKEN"]


def get_model(model_name: str) -> Any:
    """Loads a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        model: The loaded pre-trained model.

    Raises:
        ValueError: If the model_name is not supported.
    """
    if model_name == "gpt2-xl":  # in ["gpt2-xl", "gpt2"]:
        model = HookedTransformer.from_pretrained(model_name)
    elif model_name == "Llama-3.1-8B-Instruct":
        model = HookedTransformer.from_pretrained(f"meta-llama/{model_name}")
    elif model_name == "gpt2-small-sae":
        from sae_lens import SAE

        with bf.BlobFile(
            sae.SAEPath.get_path(config.VERSION, config.HOOK_ID, config.LAYER_ID),
            mode="rb",
        ) as f:
            state_dict = torch.load(f)
            model = sae.Autoencoder.from_state_dict(state_dict)
    elif model_name == "gemma-scope-2b":
        from sae_lens import SAE

        model, cfg_dict, sparsity = SAE.from_pretrained(
            release=f"{model_name}-pt-{config.TRAINED_LAYER}-canonical",
            sae_id=f"layer_{config.LAYER_ID}/width_{config.WIDTH}k/canonical",
            device=config.DEVICE,
        )
    return model


def get_tokenizer(model_name: str) -> Any:
    """Loads tokenizer.

    Args:
        model_name (str): The name of the tokenizer to load.

    Returns:
        model: The loaded tokenizer.

    Raises:
        ValueError: If the model_name is not supported.
    """
    if model_name == "gpt2-xl":  # in ["gpt2-xl", "gpt2"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_name == "Llama-3.1-8B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}")
    elif model_name == "gpt2-small-sae":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif model_name == "gemma-scope-2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding token
    return tokenizer


@torch.inference_mode()
def get_text_generator(text_generator_name: str) -> tuple:
    """Loads model and tokenizer for generating text.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.

    """
    if config.USE_API:
        if text_generator_name.startswith("gemini"):
            import google.generativeai as genai

            gemini_key = os.environ["GEMINI_KEY"]
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(text_generator_name)
        else:
            from huggingface_hub import InferenceClient

            model = InferenceClient(api_key=token)
        tokenizer = None

    else:
        # Load model with DataParallel or DistributedDataParallel
        model = AutoModelForCausalLM.from_pretrained(
            text_generator_name,
            torch_dtype=torch.float16,  # Explicitly use float16 for efficiency
            device_map="auto",  # Let the library automatically distribute across available GPUs
        )

        tokenizer = AutoTokenizer.from_pretrained(text_generator_name)
    return model, tokenizer


def get_generated_text(prompt: str, text_generator_name: str, model, tokenizer) -> str:
    """Generates text based on the given model name and prompt.

    Args:
        prompt (str): The input prompt to generate text from.
        model: Text generator model.
        tokenizer: Tokenizer for encoding text.

    Returns:
        str: The generated text response.
    """
    system_instruction = config.SYSTEM_INSTRUCTION
    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {"role": "user", "content": prompt},
    ]
    if config.USE_API:
        if text_generator_name.startswith("gemini"):
            from google.generativeai.types import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
            content = model.generate_content(prompt, safety_settings=safety_settings)
            response = content.text
        else:
            model = model.chat.completions.create(
                model=text_generator_name,
                messages=messages,
                max_tokens=100,
            )
            response = model.choices[0].message["content"]
    else:
        # Tokenize the input
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt")

        # Store original input_ids before moving to device
        input_ids = model_inputs["input_ids"]

        # Move inputs to the same device as the model
        if hasattr(model, "hf_device_map"):
            first_device = next(iter(model.hf_device_map.values()))
            model_inputs = {k: v.to(first_device) for k, v in model_inputs.items()}
        else:
            model_inputs = {k: v.to(config.DEVICE) for k, v in model_inputs.items()}

        # Generate the text samples with more configuration options
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=config.MAX_TEXT_LENGTH,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness (lower = more deterministic)
            top_p=0.95,  # Nucleus sampling parameter
            num_return_sequences=1,  # Number of different outputs to generate
        )

        # Extract just the newly generated tokens - using the stored input_ids
        generated_ids = [
            output_ids[len(input_id) :]
            for input_id, output_ids in zip(input_ids, generated_ids, strict=False)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def get_original_model(model_name: str) -> HookedTransformer:
    """Returns a pre-trained HookedTransformer model for caching activations."""
    if model_name == "gpt2-small-sae":
        model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
    elif model_name == "gemma-scope-2b":
        from sae_lens import HookedSAETransformer

        model = HookedSAETransformer.from_pretrained(
            "gemma-2-2b",
            center_writing_weights=False,
            center_unembed=False,
        )
    return model


def get_activations(
    model: Any,
    data_loader: Any,
    tensor_path: str,
    unit_id: int | None = None,
    hook_name: str = config.HOOK_NAME,
    agg_method: str = "mean",
    save: bool = True,
) -> torch.Tensor:
    """Collects activations from a specified layer of a model.

    Args:
        model: The model from which to collect activations.
        data_loader: DataLoader providing the input data.
        tensor_path (str): Path to activation tensor.
        unit_id (int): Whether to collect activations for all units or individual unit.
        hook_name (str): Name of the hook to use for collecting activations.
        agg_method (str): Aggregation method to use when applying partial_fit.
                Options: "mean" (default), "max".
        save (bool): Whether to save the activations tensor.

    Returns:
        torch.Tensor: The collected activations tensor.
    """
    # Initialize tensor to store activations
    all_agg_activations = []

    # # For SAE models, load original model at the beginning and keep it on CPU until needed
    original_model = None
    if (
        config.TARGET_MODEL_NAME == "gpt2-small-sae"
        or config.TARGET_MODEL_NAME == "gemma-scope-2b"
    ):
        original_model = get_original_model(config.TARGET_MODEL_NAME)
        original_model.to("cpu")  # Keep on CPU initially
        original_model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting activations"):
            # Clear cache before processing each batch
            torch.cuda.empty_cache()
            input_ids = batch["input_ids"].to(config.DEVICE)

            if config.TARGET_MODEL_NAME == "gpt2-small-sae":
                original_model.to(config.DEVICE)
                original_model.eval()

                logits, activation_cache = original_model.run_with_cache(
                    input_ids, remove_batch_dim=False
                )
                input_tensor = activation_cache[hook_name]

                # Move original model back to CPU to free GPU memory
                original_model.to("cpu")
                torch.cuda.empty_cache()

                layer_activations, _ = model.encode(input_tensor)
                activations = (
                    layer_activations.squeeze(1)
                    if unit_id == None
                    else layer_activations[:, :, unit_id]
                )

                # Clear cache for next iteration
                del input_tensor, layer_activations, logits, activation_cache
            elif config.TARGET_MODEL_NAME == "gemma-scope-2b":
                original_model.to(config.DEVICE)
                original_model.eval()

                logits, activation_cache = original_model.run_with_cache_with_saes(
                    input_ids, saes=model
                )
                layer_activations = activation_cache[
                    model.cfg.hook_name + ".hook_sae_acts_post"
                ]

                # Move original model back to CPU to free GPU memory
                original_model.to("cpu")
                torch.cuda.empty_cache()

                activations = (
                    layer_activations.squeeze(1)
                    if unit_id == None
                    else layer_activations[:, :, unit_id]
                )

                # Clear cache for next iteration
                del layer_activations, logits, activation_cache
            else:
                # For standard models using hooks
                layer_activations = None

                def activation_hook(act: torch.Tensor, hook: Any) -> None:  # noqa: ARG001
                    nonlocal layer_activations
                    layer_activations = act

                model.add_hook(hook_name, activation_hook)
                _ = model(input_ids)

                activations = (
                    layer_activations.squeeze(1)
                    if unit_id is None
                    else layer_activations[:, :, unit_id]
                )

                # Clear references to free memory
                del input_ids, layer_activations
                torch.cuda.empty_cache()

            # Aggregate activations based on selected method
            if agg_method == "mean":
                agg_activations = activations.mean(axis=1)
            elif agg_method == "max":
                agg_activations = activations.max(axis=1)[0]
            else:
                raise ValueError(
                    f"Unsupported aggregation method: {agg_method}. Use 'mean' or 'max'."
                )

            # Move to CPU immediately to free GPU memory
            all_agg_activations.append(agg_activations.cpu())

            # Clear references
            del activations, agg_activations
            torch.cuda.empty_cache()
            gc.collect()

        # Clean up original model if used
        if original_model is not None:
            del original_model

        # Concatenate all batch activations into a single tensor
        combined_activations = torch.cat(all_agg_activations, dim=0)

        # Save the combined activations
        if save == True:
            torch.save(combined_activations, tensor_path)

        return combined_activations
