DEVICE = "cuda"

# For model
TARGET_MODEL_NAME = (
    "gpt2-xl"
    # "gemma-scope-2b"
    # "Llama-3.1-8B-Instruct"
    # "gpt2-small-sae"
)
UNIT_ID = 6314
LAYER_ID = 0

# For evaluation
MULTI_EVAL = True  # If there are multiple descriptions per unit, set to True

if TARGET_MODEL_NAME == "gpt2-xl":
    HOOK_ID = "mlp.hook_post"
    EXPLAIN_FILE = f"{TARGET_MODEL_NAME}_layer0_layer20_layer40_60-samples.csv"
    METHOD_NAME = "GPT-explain"
elif TARGET_MODEL_NAME == "gemma-scope-2b":
    HOOK_ID = "hook_resid_post"
    EXPLAIN_FILE = "gemma-2-2b-sae-res_layer0_layer10_layer20_60-samples.csv"
    METHOD_NAME = "output-centric"
    TRAINED_LAYER = "res"
    WIDTH = 16
elif TARGET_MODEL_NAME == "gpt2-small-sae":
    HOOK_ID = "hook_resid_post"
    EXPLAIN_FILE = (
        f"{TARGET_MODEL_NAME}-resid-post_layer0_layer5_layer10_59-samples.csv"
    )
    METHOD_NAME = "output-centric"
    VERSION = "v5_32k"
elif TARGET_MODEL_NAME == "Llama-3.1-8B-Instruct":
    HOOK_ID = "mlp.hook_post"
    EXPLAIN_FILE = f"{TARGET_MODEL_NAME}_layer0_layer20_layer30_60-samples.csv"
    METHOD_NAME = "output-centric"

HOOK_NAME = f"blocks.{LAYER_ID!s}.{HOOK_ID}"
AGG_METHOD = "mean"

# For all data
STREAMING = True
BATCH_SIZE = 2  # Batch size for processing
MAX_TEXT_LENGTH = 512  # Maximum length of text excerpts

# For target data
TARGET_DATA = "c4"
DATA_FILES = None
SPLIT = "train"
SAVE_ACTIVATIONS = True

# For clustering
START_INTERVAL = 0.99
END_INTERVAL = 1.0
ACTIVATION_PERCENTILE = 90
FILTER_FOR_POSITIVE_ACTIVATIONS = True
CLUSTER_METHOD = "KMeans"
N_CLUSTERS = 5
MAX_CLUSTER_SIZE = 20
CLUSTER_N_SAMPLES = 1000
PERCENTILE_STEP = (END_INTERVAL - START_INTERVAL) / CLUSTER_N_SAMPLES
CLUSTER_EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
CLUSTER_MAX_SEQ_LEN = 8192

# For text generation
TEXT_GENERATOR_NAME = "gemini-1.5-pro"
USE_API = True
PROMPT_INSTRUCTION = (
    "You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a "
    "language model that activates in response to text excerpts. "
    'Each text starts with ">" and has a header indicated by === Text #1234 ===, where #1234 can be any number and '
    "is the identifier of the text.\n"
    "Neurons activate on a word-by-word basis. Also, neuron activations can only depend on words before the word it "
    "activates on, so the description cannot depend on words that come after, and should only depend on words that "
    "come before the activation.\n"
    "Your task is to describe what the common pattern is within the following texts. "
    "From the provided list of text excerpts, identify the concepts that trigger the activation of a particular "
    "feature. If a recurring pattern or theme emerges where these concepts appear consistently, describe this pattern."
    " Focus especially on the spans and tokens in each example that are inside a set of [delimiters] and consider the "
    "contexts they are in. The highlighted spans correspond to very important patterns.\n"
    "At the beginning, before the list of texts, there will be a list of the highlighted tokens with their activation "
    "values. "
    "At the end, following 'Description: ', your task is to write the description that fits the above "
    "criteria the best.\n"
    "Do NOT just list the highlighted words!\n"
    "Do NOT cite any words from the texts using quotation marks, but try to find overarching concepts instead!\n"
    "Do NOT write an entire sentence!\n"
    "Do NOT finish the description with a full stop!\n"
    "Do NOT mention the [delimiters] in the description!\n"
    "Do NOT include phrases like 'highlighted spans', 'Concepts of', or 'Concepts related to', "
    "and instead only state the actual semantics!\n"
    "Do NOT start with 'Description:' and instead only state the description itself!"
)
SYSTEM_INSTRUCTION = "assistant"
EVALUATION_TEXT_GENERATOR_NAME = "gemini-1.5-pro"

# For control data
CONTROL_DATA = "cosmopedia"
CONTROL_DATA_FILES = None
CONTROL_SPLIT = "train"
CONTROL_BATCH_SIZE = 64
SUBSET_SIZE = 1000

# For explain data
EXPLAIN_BATCH_SIZE = 2
N_SAMPLES = 10
PROMPT_SAMPLE_INSTRUCTION = (
    f"Generate {N_SAMPLES!s} sentences with a length of {MAX_TEXT_LENGTH!s} words, one per line, with no additional "
    f"formatting, introduction, or explanation. Each sentence should be a complete, standalone text sample that "
    f"can be saved as an individual row in a text file. The sentences should include:\n"
)
