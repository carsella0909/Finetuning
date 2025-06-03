from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from .augmentation import generate_augmentations
from .dfs_sampler import dfs_sample
from .scoring import score_candidates

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def format_prompt(self, datapoint):
        """
        Args:
            datapoint (dict): contains training data, test input
        
        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """

        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        sys = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_prompt, add_special_tokens=False)
        user = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n", add_special_tokens=False)
        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            user += inp_desc
            user += inp
            user += out_desc
            user += out

        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

        user += inp_desc
        user += self.format_grid(input_test_data)
        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)


        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        messages += assis

        return {
            "input_ids": messages,
            "input": input_test_data,
            "train": training_data
        }


    def train(self, train_dataset):
        """
        Train a model with train_dataset.
        Read a project documentation for a description of `examples` and `question`.
        """
        pass

    def predict(self, examples, questions_input):
        datapoint = {
            "train": examples,
            "test": [{"input": questions_input}]
        }

        # Generate augmentations
        augmented_data = generate_augmentations(examples, questions_input, max_augments=8)
        prompt_variants = [self.format_prompt(aug) for aug in augmented_data]
        prompt_ids_list = [torch.tensor(p['input_ids'], dtype=torch.long).tolist() for p in prompt_variants]

        # Generate candidates via DFS
        dfs_threshold = -2.3  # ~10% probability
        candidates = []
        for prompt_ids in prompt_ids_list:
            input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            dfs_outputs = dfs_sample(self.model, input_ids, self.tokenizer, threshold_logprob=dfs_threshold)
            for logp, toks in dfs_outputs:
                if toks not in candidates:
                    candidates.append(toks[len(prompt_ids):])  # remove prompt prefix

        if not candidates:
            return np.zeros_like(questions_input)  # fallback

        # Score candidates
        best_score = -float('inf')
        best_grid = None
        for cand in candidates:
            score = score_candidates(self.model, self.tokenizer, prompt_ids_list, cand)
            try:
                grid = np.array(self.parse_grid(cand))
                train_input = np.array(examples[0]['input'])
                train_output = np.array(examples[0]['output'])
                test_input = np.array(questions_input)

                if train_input.shape == train_output.shape:
                    x, y = test_input.shape
                else:
                    x = (train_output.shape[0] // train_input.shape[0]) * test_input.shape[0]
                    y = (train_output.shape[1] // train_input.shape[1]) * test_input.shape[1]

                grid = grid[:x, :y]
                if score > best_score:
                    best_score = score
                    best_grid = grid
            except:
                continue

        return best_grid if best_grid is not None else np.zeros_like(questions_input)
    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.model.load_adapter("artifacts/checkpoint-final")
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()




