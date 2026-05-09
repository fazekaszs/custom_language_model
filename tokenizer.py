import re
import pickle

from dataclasses import dataclass

# These are capital letters used in the Hungarian language.
CAPITAL_LETTERS = "ÖÜÓQWERTZUIOPŐÚASDFGHJKLÉÁŰÍYXCVBNM"

MAX_SUBSTRING_LEN = 10
SUBSTRING_FREQUENCY_THRESHOLD = 1000

@dataclass(frozen=True)
class Token:
    """
    A string wrapper, indicating that this string is already tokenized.
    """

    meaning: str

    def __repr__(self) -> str:
        return self.meaning


def search_regex(token: Token, text: list[str | Token], pattern: re.Pattern) -> list[str | Token]:
    """
    Sweeps through a mixed list of tokens and strings and looks for a specific regex inside the string parts.
    Then, its splits strings along these matches and inserts the given token in place of them.

    :param token: The token to be inserted into the list.
    :param text: The mixed list of tokens and strings.
    :param pattern: The regex pattern to look for in the strings.
    :return: The updated text with the new tokens.
    """

    output = list()
    for item in text:

        if type(item) is str:
            extension = pattern.split(item)
            for position in range(len(extension) - 1, 0, -1):
                extension.insert(position, token)
            extension = list(filter(lambda x: type(x) is Token or len(x) > 0, extension))
            output.extend(extension)
        else:
            output.append(item)

    return output


def search_capitalization(text: list[str | Token], capital_token: Token) -> list[str | Token]:
    """
    Inserts special tokens denoting that the following word contains capitalization.

    :param text: The mixed list of tokens and strings.
    :param capital_token: The token denoting next word capitalization.
    :return: The updated text with the new tokens.
    """

    output = list()
    for item in text:
        if type(item) is str:
            if set(item).intersection(set(CAPITAL_LETTERS)):
                output += [capital_token, item.lower()]
            else:
                output.append(item)
        else:
            output.append(item)

    return output


def get_substring_frequency(text: list[str | Token], substring_len: int) -> list[tuple[str, int]]:
    """
    Calculates the frequency of every length N substrings in a mixed list of tokens and strings.
    The frequency is the character coverage of the substring in the text, i.e. occurrence * N.

    :param text: The mixed list of tokens and strings.
    :param substring_len: The length of the substrings to analyze.
    :return: A sorted list of substring - frequency pairs.
    """

    frequencies = dict()
    for word in filter(lambda x: type(x) is str and len(x) >= substring_len, text):
        for idx in range(len(word) + 1 - substring_len):
            substring = word[idx:idx+substring_len]
            f = frequencies.get(substring, 0)
            frequencies[substring] = f + 1

    frequencies = [(key, substring_len * value) for key, value in frequencies.items()]
    return sorted(frequencies, key=lambda x: x[1], reverse=True)

def main():

    with open("az_egri_csillagok.txt", "r") as f:
        data = [f.read(), ]

    # Preprocessing
    data = [re.sub(r"\n+", "\n", data[0]), ]
    data = [re.sub(r" +", " ", data[0]), ]
    data = [re.sub(r"§", "", data[0]), ]

    # Defining the standard tokens
    standard_tokens_templates = [
        ("\n", r"\n"), (". ", r"\. ?"), ("! ", r"! ?"),
        ("? ", r"\? ?"), (", ", r", ?"), (": ", r": ?"),
        ("-", r"-"), ("\"", r"[\"\']"), ("(", r"\("),
        (")", r"\)"), ("; ", r"; ?"), (" ", r" ")
    ] + [(str(i), str(i)) for i in range(10)]

    # token_list contains the order of the token lookups
    token_list = list()

    # Searching for the standard token substrings and replacing them
    for meaning, pattern_str in standard_tokens_templates:
        standard_token = Token(meaning)
        token_list.append(standard_token)
        data = search_regex(standard_token, data, re.compile(pattern_str))

    # Add capitalization tokens
    capital_token = Token("<CAPITAL>")
    token_list.append(capital_token)
    data = search_capitalization(data, capital_token)

    # Print out the remaining characters present
    remaining_characters = set("".join(x for x in data if type(x) is str))
    print("\n-".join(remaining_characters))

    # Creating tokens based on the most common, variable length letter combinations
    substring_len = MAX_SUBSTRING_LEN
    while substring_len > 0:

        ssf = get_substring_frequency(data, substring_len)

        threshold_frequency = SUBSTRING_FREQUENCY_THRESHOLD if substring_len > 1 else 1
        if len(ssf) == 0 or ssf[0][1] < threshold_frequency:
            substring_len -= 1
            continue

        print(f"Winner word: {ssf[0][0]} with frequency {ssf[0][1]}")
        word_token = Token(ssf[0][0])
        token_list.append(word_token)
        data = search_regex(word_token, data, re.compile(ssf[0][0]))

    # Replace tokens only with their indices
    data = [token_list.index(x) for x in data]

    with open("tokenized_text.pickle", "wb") as f:
        pickle.dump({"tokenized_text": data, "token_list": token_list}, f)


if __name__ == '__main__':
    main()
