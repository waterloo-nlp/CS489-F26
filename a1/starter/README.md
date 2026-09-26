# CS 489 Assignment 1: starter code

Copy every file in this directory into your own project root (next to your
`part1.1-clean.py`, `part2.py`, and so on). The files marked **do not
modify** below are fixed: do not modify or submit them, and the TAs restore
pristine copies of them before re-running any entry point. Your implemented
`part2_tokenizer.py` and your extended `pyproject.toml` (with `uv.lock`)
are part of your submission.

## Contents

| File | Role | Status |
|------|------|--------|
| `part2_transformer_lm.py` | the fixed Part-2 GPT-style Transformer | **do not modify** |
| `part2_train.py` | the fixed Part-2 trainer | **do not modify** |
| `part2_eval_bpc.py` | the fixed bits-per-character evaluator | **do not modify** |
| `part3_artifacts.py` | save/load helpers defining the Part-3 model format | **do not modify** |
| `data.py` | corpus and token-id IO helpers | **do not modify** |
| `part2_tokenizer.py` | Part-2 tokenizer interface | **stub: you implement it** |
| `part2_config.example.json` | annotated template for `artifacts/part2/config.json` | template |
| `pyproject.toml` | minimal `uv` project definition | extend and lock yourself |

Names carry the assignment part each file serves (`part2_*` for Part 2,
`part3_artifacts.py` for Part 3); `data.py` and `pyproject.toml` are shared
across parts.

Note what is deliberately **not** here: anything that trains, encodes, or
decodes BPE or unigram tokenizers. Tokenization is the assignment; the
starter only provides the fixed model/trainer/evaluator, the artifact
formats, and small IO helpers.

## Environment

Use the provided `pyproject.toml` as a starting point, add anything else you
need, then run `uv lock` and commit both `pyproject.toml` and `uv.lock`.
Every entry point must run as `uv run python <script>` from a fresh
`uv sync`. Python 3.11 or later is required.

## Part-2 workflow

The trainer consumes **token-id streams**, not text, so that it works
unchanged with any tokenizer. Token-id files are uint32 binary
(`data.write_ids` / `data.read_ids`), which is why V <= 2^32.

Your `part2.py` should orchestrate the following (you may also run the
steps manually while experimenting):

1. Train your chosen tokenizer at the requested vocabulary size on the
   Part-1 cleaned training text (`data.load_split(input_dir, "train")`
   gives you the canonical concatenation).
2. Encode the cleaned train and dev texts and write `train.bin` /
   `dev.bin` with `data.write_ids`.
3. Train the fixed model:

   ```bash
   uv run python part2_train.py --train-ids train.bin --dev-ids dev.bin \
       --vocab-size 50000 --output artifacts/part2/ \
       --dev-text <path to the concatenated cleaned dev text>
   ```

   This writes `artifacts/part2/model.pt` and `artifacts/part2/run_config.json`
   and prints the final dev NLL, token perplexity, and (with `--dev-text`)
   dev bits-per-character. On the Colab free-tier T4 one run takes on the
   order of 15 to 30 minutes.
4. Copy `run_config.json` to `artifacts/part2/config.json` and add your
   `tokenizer` section (see below and `part2_config.example.json`).
5. Save your tokenizer files under `artifacts/part2/tokenizer/` and
   implement `part2_tokenizer.load_tokenizer` (see the stub's docstring).
6. For every bpc number you report, use the fixed evaluator:

   ```bash
   uv run python part2_eval_bpc.py --config artifacts/part2/config.json \
       --model artifacts/part2/model.pt --ids dev.bin --text <cleaned dev text>
   ```

## Part-2 config format

`artifacts/part2/config.json` (JSON or YAML) must contain:

- `vocab_size` (int): your tokenizer's vocabulary size V.
- `model` (object): the architecture hyperparameters expected by
  `part2_transformer_lm.py`: `context_length`, `d_model`, `n_layer`, `n_head`,
  `d_ff`, `dropout`. These are fixed by `part2_train.py`; copy them from the
  generated `run_config.json`.
- `training` (object): the fixed training hyperparameters, also copied from
  `run_config.json`.
- `tokenizer` (object): `algorithm` (for example `"bpe"` or `"unigram"`)
  and a `settings` object with everything needed to understand your
  tokenization choice.

The TAs rebuild your checkpoint with
`part2_transformer_lm.load_checkpoint(config, model.pt)`, which reads only
`vocab_size` and `model`, and score it with `part2_tokenizer.load_tokenizer`
plus the `part2_eval_bpc.py` logic on the held-out test corpus.

## Bits-per-character definition

Identical for dev and test, implemented by `part2_eval_bpc.py`:

- `N_chars` is `len(text)` in Unicode code points (`data.count_chars`).
- A window of size `context_length + 1` slides over the token-id stream
  with stride `context_length`; every token after the first is scored,
  including the trailing partial window. Each scored token is predicted from
  its true preceding context. The first token has no preceding context and
  is not scored.
- `bpc = (sum of token-level NLL in nats) / (N_chars * ln 2)`.

Your tokenizer's encoding must be lossless: decoding its token IDs must
reproduce the input text exactly, including line breaks, repeated whitespace,
and unseen characters (`decode(encode(text)) == text`). Otherwise this
comparison is not meaningful.

## Part-3 model artifact

Serialize your selected n-gram model with
`part3_artifacts.save_ngram_model(model, "artifacts/part3/model.json.gz")`.
The required format (count tables for every order 1..n, space-joined
token-id keys, add-alpha scoring semantics) is documented in the
`part3_artifacts.py` module docstring; `part3_artifacts.ngram_key` builds the keys.
Models are trained and scored line by line: n-grams never cross line
boundaries, the EOS token (id 50,000) is appended to every line and scored
like any other token, and an optional BOS token (id 50,001) may be
prepended if you record it in the model file. The recorded `vocab_size`
must include these special tokens (50,001 with EOS, 50,002 with EOS and
BOS). The TAs load the file with `part3_artifacts.load_ngram_model` and score it
with your Part-1 BPE tokenizer. Full high-order count tables can exceed 100 MB
gzipped, so keep the submission size in mind when you select `(n, alpha)`.

## Practical notes

- Reproducibility: `part2_train.py` fixes the seed (489), but GPU floating-point
  reductions are not bitwise deterministic, so small run-to-run variation
  in the last decimal places is normal.
- The provided model uses about 30M parameters at V = 50,000 and fits
  comfortably in the 16 GB memory of a Colab T4.
- If `uv` is unavailable in your Colab session, install it first per the
  uv documentation, then `uv sync`.
