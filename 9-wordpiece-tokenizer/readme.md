This is a comparison between Wordpiece and Byte pair encoding tokenization.


$$WP = f(AB)/(f(A)f(B))$$
$$BPE = f(AB)$$

$WP$ is the scoring function of Wordpiece

$BPE$ is the scoring function of Byte Pair Encoding

At each iteration the tokenizer merges the top tokens according to the scoring function applied to the vocabulary

Byte Pair Encoding just merges the most frequently occuring pairs together.

Wordpiece gives priority to merge into the vocab rare tokens that occur together often.

How to run : 
``` 
python wp_versus_bpe.py
```

___

## Description

This Python script demonstrates two tokenization techniques, Byte Pair Encoding (BPE) and WordPiece, applied to a given training corpus. These techniques are used in natural language processing (NLP) to reduce the size of the vocabulary and handle unknown words more gracefully.

The script starts by importing necessary modules and defining several functions:

explode_word: Splits a word into characters, with all characters after the first one prefixed with "##". This is a common practice in NLP to differentiate the starting character of a word from its continuation.
encode: Applies explode_word to each word in the training corpus, effectively converting the corpus into a list of lists, where each inner list contains the exploded form of a word.
make_bigrams: Generates bigrams (pairs of adjacent tokens) for each word in the encoded corpus.
generate_merge_token: Creates a string representation of a potential merge between two tokens, which is used for scoring and identifying merge candidates.
freq_map: Calculates the frequency of each token in the encoded training corpus.
get_a_b_ab: Parses a merge token string to extract the original tokens (a and b) and their merged form (ab).
bpe_scoring and wordpiece_scoring: Define scoring functions for BPE and WordPiece tokenization methods, respectively. BPE scoring is based on the frequency of the merge candidate, while WordPiece scoring considers the frequency of the merge candidate relative to the frequencies of the original tokens.
tokenization: The main function that implements the tokenization process. It encodes the training corpus, initializes the vocabulary, and iteratively merges tokens based on the scoring function. After each merge, it updates the encoded corpus and prints the top merge candidate.
The script concludes with an example usage section, where it tokenizes a sample training corpus using both BPE and WordPiece methods. It demonstrates the iterative process of merging tokens to reduce the vocabulary size, which is crucial for efficient NLP model training and inference.

This script is a simplified demonstration of BPE and WordPiece tokenization, focusing on the core concepts and mechanics of these techniques without delving into optimizations or integration with larger NLP systems.