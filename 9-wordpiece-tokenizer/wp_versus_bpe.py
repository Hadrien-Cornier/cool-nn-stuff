from collections import defaultdict
from typing import Callable, List, Tuple, Dict

def explode_word(word: str) -> List[str]:
    return [word[0]] + ["##" + word[i] for i in range(1, len(word))]

def encode(training_corpus: str) -> List[List[str]]:
    return [explode_word(word) for word in training_corpus.split()]

def make_bigrams(L: List[List[str]]) -> List[List[Tuple[str, str]]]:
    return [list(zip(sub, sub[1:])) for sub in L]

def generate_merge_token(p: str, s: str) -> str:
    if p.startswith("##"):
        return p + "+" + s + " = ##" + (p + s).replace("#", "")
    return p + "+" + s + " = " + p + s.replace("#", "")

def freq_map(encoded_training_corpus: List[List[str]]) -> Dict[str, int]:
    freq_map = defaultdict(int)
    for word in encoded_training_corpus:
        for token in word:
            freq_map[token] += 1
    return freq_map

def get_a_b_ab(code: str) -> Tuple[str, str]:
    return code.split("+")[0], code.split("+")[1].split(" = ")[0], code.split("=")[1].strip()

def bpe_scoring(merge_token: str, freq: int, token_freq: Dict[str, int]) -> float:
    return freq

def wordpiece_scoring(merge_token: str, freq: int, token_freq: Dict[str, int]) -> float:
    a, b, _ = get_a_b_ab(merge_token)
    return freq / (token_freq[a] * token_freq[b])

def tokenization(training_corpus: str, num_steps: int, scoring_function: Callable[[str, int, Dict[str, int]], float]):
    E = encode(training_corpus)
    vocab = set().union(*[set(word) for word in E])

    def apply_merge(merge: str, E: list) -> List[List[str]]:
            a,b,ab = get_a_b_ab(merge)
            new_E = []
            for word in E:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i] == a) and (word[i + 1] == b):
                        new_word.append(ab)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_E.append(new_word)
            return new_E

    for step in range(num_steps):
        token_freq = freq_map(E)
        tuples = make_bigrams(E)
        
        SF = defaultdict(int)
        for W in tuples:
            for B in W:
                SF[generate_merge_token(*B)] += 1
        
        scores = [(merg, scoring_function(merg, freq, token_freq)) for merg, freq in SF.items()]
        scores = sorted(scores, key=lambda x: -x[1])
        
        top_merge = scores[0]
        print(f"Step {step + 1}: Top merge candidate: {top_merge}")

        new_E = apply_merge(top_merge[0], E)
        print("we merged", top_merge[0], " in ", E[:10] , "and got", new_E[:10])
        
        E = new_E
        
        # Here you would implement the actual merging logic
        # For simplicity, we're just printing the top merge candidate
        
        # Update E and vocab based on the merge (not implemented in this summary)

    return vocab

# Example usage
training_corpus = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"
num_steps = 20

print("Byte Pair Encoding Tokenization:\n\n")
bpe_vocab = tokenization(training_corpus, num_steps, bpe_scoring)

print("\n\nWordPiece Tokenization:\n\n")
wordpiece_vocab = tokenization(training_corpus, num_steps, wordpiece_scoring)