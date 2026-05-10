# Implementing a Small Language Model

## How was this code created and what is its purpose?

This repo contains the implementation of a small, not so useful language model that was only meant 
to serve as a fun coding exercise.
Although several concepts I used are already pre-implemented in PyTorch, I chose to code most of 
the building blocks myself.
I chose to mostly lean on my accumulated knowledge or things I am able to look up in documentations and
tutorials, while using only a minimal amount of LLM support (only for quality control and correctness
validation).
__Nothing is vibe-coded in this repo!__
This implies that there can be errors and/or unconventional solutions in the code, so proceed 
with caution.

## How is the language model structured?

The model is an autoregressive, deep, masked transformer with rotary positional encoding (RoPE).
The training text is tokenized with the following strategy:

1. the text is preprocessed a little, replacing multiple spaces and new line characters with 
   single ones,
2. some standard characters, like spaces, new line characters, periods etc. are tokenized first,
3. `<CAPITAL>` tokens are added to indicate word capitalization,
4. next, all length _N_ substrings are collected from the yet untokenized parts and ordered 
   according to their frequencies. The most frequent substring is tokenized if its frequency is
   above a threshold. This statistics is recalculated and the new most frequent substring follows
   the process. If the frequency of the most frequent substring is not above the threshold, the
   substring length _N_ is decreased by one.
5. Single letters (_N_ = 1) are tokenized with a threshold frequency of 1, so at the end, the whole
   text is tokenized.

This process is implemented in `tokenizer.py`, which outputs a pickled file `tokenized_text.pickle`.
This file is a dictionary containing the list of unique tokens and the full text in the form of a
token index list.
This file is read by the `transformer.py` script that contains the heavy stuff.
It implements the language model under several layers of abstraction:

- the `TokenEmbedding` module is able to convert a batch of token index sequences to a batch of
  embedding sequences,
- the `ScaledDotProductAttention` (SDPA) module naively implements a KQV masked attention mechanism,
- the `MultiHeadAttention` (MHA) module contains several SDPA heads with the output concatenation 
  strategy,
- the `TransformerBlock` combines a MHA module and a 2-layer feedforward module with _GeLU_ 
  activation in between, all with residual connections to the main tensor stream,
- the `DeepTransformer` is just a sequence of transformer blocks chaid together,
- the `LanguageModel` combines all these; it starts with the token embedding module, then with
  the deep transformer module, and finishes with a singe linear layer. Note that the outputs are
  logits, rather than probability mass functions, i.e. there isn't a final _SoftMax_ activation.

The following hyperparameters were chosen (without giving it much thought):

|            hyperparameter name            |   hyperparameter value    |
|:-----------------------------------------:|:-------------------------:|
|            embedding dimension            |            128            |
|         query and key dimensions          |             8             |
|              value dimension              |             8             |
|    number of SDPA heads per MHA module    |             5             |
| MLP hidden layer dimension and activation | 2 * value dimension, GeLU |
|                model depth                |             4             |
|                batch size                 |            256            |
|               learning rate               |           1E-3            |

With these settings, the model has 427,830 parameters in total.
Matrix parameters for the embedding and SDPA layers were initialized from a normal distribution
with 0 mean and a standard deviation of `1 / sqrt(dim)`.

## What is it trained on and how is it trained?

The model is trained on a tiny corpus of Hungarian text, specifically on the book called 
_Egri csillagok_ from _Géza Gárdonyi_.
It was downloaded from the website of the 
[Magyar Elektronikus Könyvtár](https://mek.oszk.hu/00600/00656/).
The whole book was copied to a single txt file and some rare characters and substring (like `=` or
numbers between square brackets) were removed.
After tokenization, a vocabulary size of 310 tokens was achieved with the main text containing 
536,238 tokens.
The training loss was a weighted cross entropy loss, where the weights are the inverse square roots
of token occurrences.
A batch size of 256 was used with a context size of 100 (or, rather, 99, due to autoregression).
Since all overlapping windows were used, this meant that an epoch consisted of 2095 batches.

## How is training monitored?

Training was followed through the loss and accuracy metrics.
Accuracy was not weighted.
These loss and accuracy curves were plotted after every epoch.
The model was also saved as a `.pt` file after every epoch.
Finally, also after every epoch, a text sentence is generated (deterministically) starting
from the seed text `<CAPITAL>|gergely| |folytatta|:` (token indices `[22, 28, 11, 23, 5, ]`,
tokens are separated with a `|` character for clarity).
The deterministic generation corresponds to a temperature 0 generation, i.e. by selecting the
most probable next token.
Here is an example plot of the first epoch:

<div style="text-align: center;">
    <img src="https://github.com/fazekaszs/custom_language_model/blob/master/images/epoch0.png" alt="training" width=800/>
</div>

## Notable used and implemented features

- tokenization implementation,
- PyTorch's built-in `Dataset` and `DataLoader` classes,
- embedding layer implementation,
- rotary positional encoding (RoPE) implementation, 
- PyTorch's built-in `LayerNorm` layer,
- SDPA, MHA and transformer layer implementations,
- PyTorch's built-in `Adam` optimizer,
- optional training on CUDA compatible GPUs,
- (weighted) cross entropy loss and accuracy monitoring,
- checkpoint `.pt` file writing,
- switching between training and evaluation modes

## Planned additions

- inclusion of test and validation sets,
- learning rate scheduling,
- weight decay,
- forward and backward hooks for monitoring activations and gradients,
- ablation studies and hyperparameter optimization,
- optional gating incorporation into the SDPA module