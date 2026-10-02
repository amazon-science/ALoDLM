# Modified for ALoDLM: adaptive recurrence, depth-aware caching, and release packaging.
# Original notices are retained; see NOTICE and licenses/WeDLM.txt.

# coding=utf-8
"""Hardware-agnostic FLOP accounting for the looped engine.

Unit = token-layer-calls: one transformer layer applied to one stream token once.
Every executed loop pass adds stream_len x layers_run, so adaptive-depth early stops,
sandwich coda branches, and window recomputation are all captured by construction.
FLOP/token ~= 2 * P_layer_avg * token_layer_calls / committed_tokens (+ embed/head terms),
derived offline from the model's known parameter counts.

Module-level accumulators (engine worker process); snapshot flows out via
llm_engine global_stats -> wedlm_eval metrics.json.
"""

# loop_passes counts EVERY layer-group execution -- prelude, looped block, coda branches -- so it
# is a FLOP accumulator, not a depth counter. Dividing it by forwards gave 10.03 on a K=8 run and
# 0.75 on a K=4 one. loop_block_passes counts only the looped block, which is what "average loop
# number" means, and pairs with forward_calls (batched forwards, same population).
_S = {"token_layer_calls": 0, "loop_passes": 0, "loop_block_passes": 0,
      "forward_calls": 0, "stream_tokens": 0}


def add_pass(stream_tokens: int, layers: int, loop_block: bool = False) -> None:
    _S["token_layer_calls"] += stream_tokens * layers
    _S["loop_passes"] += 1
    if loop_block:
        _S["loop_block_passes"] += 1


def add_forward(stream_tokens: int) -> None:
    _S["forward_calls"] += 1
    _S["stream_tokens"] += stream_tokens


def snapshot() -> dict:
    return dict(_S)


def reset() -> None:
    for k in _S:
        _S[k] = 0
