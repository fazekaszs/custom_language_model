import pickle
import math

from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as tofu

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from tokenizer import Token


class TextDataset(Dataset):
    """
    Stores the training data and handles token list slicing.
    """

    def __init__(self, tokenized_text: list[int], context_size: int) -> None:
        self.context_size = context_size
        self.tokenized_text = tokenized_text

    def __len__(self):
        return len(self.tokenized_text) + 1 - self.context_size

    def __getitem__(self, idx: int):
        return np.array(self.tokenized_text[idx:idx+self.context_size], dtype=int)


class TokenEmbedding(nn.Module):
    """
    Converts a list of tokens to a batch of embedding matrices.
    """

    def __init__(
        self,
        n_tokens: int,
        embedding_dim: int,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_mx = nn.Parameter(torch.normal(
            torch.zeros((n_tokens, embedding_dim)),
            1. / math.sqrt(embedding_dim)
        ))

    def forward(self, x: list[np.ndarray]) -> torch.Tensor:

        output = list()
        for sample_indices in x:
            output.append(self.embedding_mx[sample_indices])
        return torch.stack(output)


class ScaledDotProductAttention(nn.Module):
    """
    Implements a vanilla, masked SDPA block with RoPE.
    """

    def __init__(
        self,
        embedding_dim: int,
        kq_dim: int,
        v_dim: int,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.key_matrix = nn.Parameter(torch.normal(
            torch.zeros((embedding_dim, kq_dim)),
            1. / math.sqrt(kq_dim)
        ))
        self.query_matrix = nn.Parameter(torch.normal(
            torch.zeros((embedding_dim, kq_dim)),
            1. / math.sqrt(kq_dim)
        ))
        self.value_matrix = nn.Parameter(torch.normal(
            torch.zeros((embedding_dim, v_dim)),
            1. / math.sqrt(kq_dim)
        ))

        self.rope_theta_matrices = None
        self.rope_sin_mask = None
        self.rope_sin_signs = None

    def update_rope_theta_matrices(
        self,
        contex_len: int,
        embedding_dim: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> None:
        """
        Recalculates the RoPE cos(i * theta_j) and sin(i * theta_j) matrices used for the RoPE
        transformation and caches them. This must be updated every time the context length changes.

        :param contex_len: The dimension along which the embeddings are rotated.
        :param embedding_dim: The dimension along which the rotation frequency changes.
        :param dtype: The dtype of the matrix to be transformed.
        :param device: The device the matrix to be transformed is on.
        """

        if embedding_dim % 2 != 0:
            raise Exception("The embedding dimension must be even for RoPE to work!")

        # context_support will have a shape of (context_len, )
        context_support = torch.arange(contex_len, dtype=dtype, device=device)

        # repeat_interleave converts theta to [a, b, ...] -> [a, a, b, b, ...]
        # after that, it will have a shape of (embedding_dim, )
        theta = torch.arange(embedding_dim // 2, dtype=dtype, device=device)
        theta = 10000. ** (-2. * theta / embedding_dim)
        theta = torch.repeat_interleave(theta, 2)

        # angle_matrix will have a shape of (1, context_len, embedding_dim)
        angle_matrix = context_support[None, :, None] * theta[None, None, :]

        # set the elements of the cos(i * theta_j) and sin(i * theta_j) matrices
        self.rope_theta_matrices = (torch.cos(angle_matrix), torch.sin(angle_matrix))

    def update_rope_sin_components(
        self,
        embedding_dim: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> None:
        """
        Updates the permutation mask and sign multiplier matrices applied to the sin component of RoPE,
        and caches them. This is independent of the context length, so it must be calculated only once.

        :param embedding_dim: The dimension along which the rotation frequency changes.
        :param dtype: The dtype of the matrix to be transformed.
        :param device: The device the matrix to be transformed is on.
        :return:
        """

        self.rope_sin_mask = torch.arange(embedding_dim, dtype=torch.int64, device=device)
        self.rope_sin_mask[::2] += 1
        self.rope_sin_mask[1::2] -= 1

        self.rope_sin_signs = torch.ones(embedding_dim, dtype=dtype, device=device)
        self.rope_sin_signs[1::2] = -1.

    def rope_transform(self, embedding: torch.Tensor) -> torch.Tensor:

        # embedding has a shape of (batch_size, context_len, embedding_dim)

        if self.rope_theta_matrices is None or self.rope_theta_matrices[0].shape[1] != embedding.shape[1]:
            self.update_rope_theta_matrices(embedding.shape[1], embedding.shape[2], embedding.dtype, embedding.device)

        if self.rope_sin_mask is None:
            self.update_rope_sin_components(embedding.shape[2], embedding.dtype, embedding.device)

        cos_theta, sin_theta = self.rope_theta_matrices
        embedding = embedding * cos_theta + self.rope_sin_signs * embedding[:, :, self.rope_sin_mask] * sin_theta

        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x has a shape of (batch_size, context_len, embedding_dim)

        keys = x @ self.key_matrix
        queries = x @ self.query_matrix
        values = x @ self.value_matrix

        keys = self.rope_transform(keys)
        queries = self.rope_transform(queries)

        score_logits = torch.einsum("biq,bjq->bij", queries, keys)
        score_logits /= math.sqrt(keys.shape[2])
        mask = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
        score_logits[:, *mask] = float("-inf")
        scores = tofu.softmax(score_logits, dim=2)

        aggregated_values = torch.einsum("bij,bjv->biv", scores, values)

        return aggregated_values


class MultiHeadAttention(nn.Module):
    """
    Collects several SDPA blocks into a multi head attention block.
    """

    def __init__(
        self,
        n_heads: int,
        embedding_dim: int,
        kq_dim: int,
        v_dim: int,
        *args, **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.heads = list()
        for idx in range(n_heads):
            current_head = ScaledDotProductAttention(embedding_dim, kq_dim, v_dim)
            self.add_module(f"head{idx}", current_head)
            self.heads.append(current_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x has a shape of (batch_size, context_len, embedding_dim)

        output = list()
        for head in self.heads:
            output.append(head(x))

        return torch.concatenate(output, dim=2)


class TransformerBlock(nn.Module):
    """
    Expands a multi head attention block to a transformer block with layer norms and MLPs.
    """

    def __init__(
        self,
        n_heads: int,
        embedding_dim: int,
        kq_dim: int,
        v_dim: int,
        *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        self.mha_ln = nn.LayerNorm([embedding_dim, ])
        self.mha = MultiHeadAttention(n_heads, embedding_dim, kq_dim, v_dim)
        self.projector = nn.Linear(n_heads * v_dim, embedding_dim)

        self.mlp_ln = nn.LayerNorm([embedding_dim, ])
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Linear(2 * embedding_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual_stream = self.mha(self.mha_ln(x))
        residual_stream = self.projector(residual_stream)
        x = x + residual_stream

        residual_stream = self.mlp(self.mlp_ln(x))
        x = x + residual_stream

        return x


class DeepTransformer(nn.Module):
    """
    Chains transformer blocks sequentially together into a deep transformer.
    """

    def __init__(
        self,
        depth: int,
        n_heads: int,
        embedding_dim: int,
        kq_dim: int,
        v_dim: int,
        *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        self.blocks = list()

        for idx in range(depth):

            current_block = TransformerBlock(n_heads, embedding_dim, kq_dim, v_dim)
            self.add_module(f"TransformerBlock{idx}", current_block)
            self.blocks.append(current_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for transformer_block in self.blocks:
            x = transformer_block(x)

        return x


class LanguageModel(nn.Module):
    """
    Implementation of the small language model from embedding, through the deep transformer module,
    ending with the unembedding layer.
    """

    def __init__(
        self,
        n_tokens: int,
        depth: int,
        n_heads: int,
        embedding_dim: int,
        kq_dim: int,
        v_dim: int,
        *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        self.embedding_layer = TokenEmbedding(n_tokens, embedding_dim)
        self.deep_transformer = DeepTransformer(
            depth,
            n_heads,
            embedding_dim,
            kq_dim,
            v_dim
        )
        self.unembed_layer = nn.Linear(embedding_dim, n_tokens)

    def forward(self, x: list[np.ndarray]) -> torch.Tensor:

        tensor_stream = self.embedding_layer(x)
        tensor_stream = self.deep_transformer(tensor_stream)
        return self.unembed_layer(tensor_stream)


def test_sentence(token_list: list[Token], language_model: LanguageModel, device: torch.device):

    sentence_indices = [22, 28, 11, 23, 5, ]
    for idx in range(50):

        sentence_tensor = torch.tensor([sentence_indices, ], dtype=torch.int64, device=device)

        with torch.no_grad():
            logits = language_model(sentence_tensor)
            selected_token_idx = np.argmax(logits.cpu().numpy(), axis=2)
            sentence_indices.append(selected_token_idx[0, -1])

    sentence_str = ""
    for idx in sentence_indices:
        sentence_str = sentence_str + token_list[idx].meaning + "|"

    return sentence_str


def main():

    with open("tokenized_text.pickle", "rb") as f:
        tokenization: dict[str, Any] = pickle.load(f)

    tokenized_text: list[int] = tokenization["tokenized_text"]
    token_list: list[Token] = tokenization["token_list"]

    print(f"Vocabulary size: {len(token_list)} tokens")
    print(f"Full text length: {len(tokenized_text)} tokens")

    text_dataset = TextDataset(tokenized_text, 100)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    token_frequencies = list()
    for idx in range(len(token_list)):
        frequency = np.sum(np.array(tokenized_text) == idx) / len(tokenized_text)
        token_frequencies.append(frequency)
    token_frequencies = torch.tensor(token_frequencies, dtype=torch.float, device=device)

    language_model = LanguageModel(
        n_tokens=len(token_list),
        depth=4,
        n_heads=5,
        embedding_dim=128,
        kq_dim=8,
        v_dim=8,
    ).to(device)

    n_trainable_params = sum(p.numel() for p in language_model.parameters() if p.requires_grad)
    print(f"Number of trainable model parameters: {n_trainable_params}")

    optimizer = Adam(language_model.parameters(), lr=1E-3)

    for epoch in range(100):

        text_data_loader = DataLoader(text_dataset, batch_size=256, shuffle=True)
        epoch_losses = list()
        epoch_accuracies = list()
        for element_idx, element in enumerate(text_data_loader):

            optimizer.zero_grad()

            element_in = element[:, :-1]
            element_out = element[:, 1:].to(device)
            predicted_out = language_model(element_in)
            loss = tofu.cross_entropy(
                torch.permute(predicted_out, (0, 2, 1)),
                element_out,
                weight=1 / torch.sqrt(token_frequencies)
            )

            epoch_losses.append(float(loss.detach().cpu()))

            predicted_classes = np.argmax(
                predicted_out.detach().cpu().numpy(),
                axis=2
            ).flatten()
            accuracy = accuracy_score(element_out.cpu().numpy().flatten(), predicted_classes)
            epoch_accuracies.append(accuracy)

            loss.backward()
            optimizer.step()

            mean_loss = np.mean(epoch_losses)
            mean_accuracy = np.mean(epoch_accuracies)

            print(
                f"\repoch = {epoch}, "
                f"item {element_idx}/{len(text_data_loader)}, "
                f"mean loss = {mean_loss:.3f}, "
                f"mean accuracy = {mean_accuracy:.3%}",
                end=""
            )

        print()

        # Plot loss curve for the current epoch
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)

        ax[0].plot(epoch_losses, color="black")
        ax[0].set_xlabel("batch index")
        ax[0].set_ylabel("weighted cross entropy loss")
        ax[0].set_yscale("log")

        ax[1].plot(epoch_accuracies, color="black")
        ax[1].set_xlabel("batch index")
        ax[1].set_ylabel("batch accuracy")
        ax[1].set_ylim(0, 1)

        fig.savefig(f"epoch{epoch}.png", dpi=300)

        # Save model
        torch.save(language_model.state_dict(), f"epoch{epoch}_checkpoint.pt")

        # Print out a test sentence
        language_model.eval()
        print("Test sentence: ", test_sentence(token_list, language_model, device))
        language_model.train()


if __name__ == "__main__":
    main()
