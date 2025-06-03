import numpy as np
import random
import itertools
from copy import deepcopy


def rotate_grid(grid, k):
    return np.rot90(grid, k).tolist()

def flip_grid(grid, axis):
    arr = np.array(grid)
    return (np.flipud(arr) if axis == 0 else np.fliplr(arr)).tolist()

def permute_colors(grid, permutation):
    arr = np.array(grid)
    result = np.vectorize(lambda x: permutation.get(x, x))(arr)
    return result.tolist()

def shuffle_examples(examples):
    return random.sample(examples, len(examples))

def generate_augmentations(examples, test_input, max_augments=8):
    """
    Generate augmented versions of the training examples and test input.

    Args:
        examples: list of {"input": grid, "output": grid} dicts
        test_input: input grid for test
        max_augments: how many augmented versions to generate

    Returns:
        List of dicts {"train": [...], "test": [test_dict]}
    """
    augmentations = []

    d8_transforms = [
        lambda g: g,
        lambda g: rotate_grid(g, 1),
        lambda g: rotate_grid(g, 2),
        lambda g: rotate_grid(g, 3),
        lambda g: flip_grid(g, 0),
        lambda g: flip_grid(g, 1),
        lambda g: rotate_grid(flip_grid(g, 0), 1),
        lambda g: rotate_grid(flip_grid(g, 1), 1),
    ]

    color_values = list(range(10))
    perms = random.sample(list(itertools.permutations(color_values)), max_augments)

    for i in range(max_augments):
        T = d8_transforms[i % len(d8_transforms)]
        color_perm = {k: v for k, v in zip(color_values, perms[i])}

        aug_examples = []
        for pair in examples:
            inp = T(permute_colors(pair['input'], color_perm))
            out = T(permute_colors(pair['output'], color_perm))
            aug_examples.append({"input": inp, "output": out})

        aug_test = {"input": T(permute_colors(test_input, color_perm))}

        aug_examples = shuffle_examples(aug_examples)

        augmentations.append({"train": aug_examples, "test": [aug_test]})

    return augmentations
