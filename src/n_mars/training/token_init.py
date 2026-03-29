"""UNDO token initialization strategies for N-MARS.

Three methods for initializing the <UNDO> token embedding, as described
in the N-MARS paper Appendix.

Usage:
    from n_mars.training.token_init import initialize_undo_token

    model, tokenizer = initialize_undo_token(model, tokenizer, method="semantic")
"""

from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

UNDO_TOKEN = "<UNDO>"

# Anchor words for semantic initialization
SEMANTIC_ANCHOR_WORDS = [
    "delete",
    "remove",
    "undo",
    "erase",
    "back",
    "cancel",
    "retry",
    "revert",
    "reset",
    "clear",
    "backspace",
]


def _init_centroid(model: PreTrainedModel) -> torch.Tensor:
    """Initialize as the mean of all vocabulary embeddings."""
    embeddings = model.get_input_embeddings().weight.data
    return embeddings.mean(dim=0)


def _init_context(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Initialize from the last hidden state when encoding the description."""
    desc = "this token is used to delete the previous token in the response"
    inputs = tokenizer(desc, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Use last token of last hidden layer
    hidden = outputs.hidden_states[-1][:, -1, :]
    return hidden.squeeze(0).float()


def _init_semantic(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Initialize as the mean of anchor word embeddings."""
    embeddings = model.get_input_embeddings().weight.data
    anchor_ids = []
    for word in SEMANTIC_ANCHOR_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            anchor_ids.append(ids[0])
        else:
            logger.warning("Anchor word %r not found in tokenizer vocabulary", word)

    if not anchor_ids:
        raise ValueError("No anchor words found in tokenizer vocabulary")

    anchor_tensor = torch.tensor(anchor_ids, device=embeddings.device)
    return embeddings[anchor_tensor].mean(dim=0)


def initialize_undo_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    method: str = "semantic",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Add <UNDO> as a special token and initialize its embedding.

    Args:
        model: A causal LM model.
        tokenizer: Corresponding tokenizer.
        method: One of "centroid", "context", or "semantic".

    Returns:
        (model, tokenizer) with the new token registered and embeddings resized.
    """
    if method not in ("centroid", "context", "semantic"):
        raise ValueError(f"Unknown method {method!r}; choose 'centroid', 'context', or 'semantic'")

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add <UNDO> as a special token
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [UNDO_TOKEN]})
    logger.info("Added %d new special token(s): %s", num_added, UNDO_TOKEN)

    undo_token_id = tokenizer.convert_tokens_to_ids(UNDO_TOKEN)
    logger.info("<UNDO> token id: %d", undo_token_id)

    # Compute initialization vector BEFORE resizing (vocab is still old)
    if method == "centroid":
        init_vec = _init_centroid(model)
    elif method == "context":
        init_vec = _init_context(model, tokenizer)
    else:  # semantic
        init_vec = _init_semantic(model, tokenizer)

    # Resize embeddings to accommodate the new token
    model.resize_token_embeddings(len(tokenizer))

    # Initialize input embedding for <UNDO>
    with torch.no_grad():
        model.get_input_embeddings().weight[undo_token_id] = init_vec.to(
            model.get_input_embeddings().weight.dtype
        )
        logger.info("Initialized input embedding for <UNDO> via method=%s", method)

        # Initialize output projection (lm_head) if separate from input embeddings
        output_embeddings = model.get_output_embeddings()
        input_emb_ptr = model.get_input_embeddings().weight.data_ptr()
        if output_embeddings is not None and output_embeddings.weight.data_ptr() != input_emb_ptr:
            output_embeddings.weight[undo_token_id] = init_vec.to(output_embeddings.weight.dtype)
            logger.info("Initialized output projection (lm_head) for <UNDO>")

    return model, tokenizer
