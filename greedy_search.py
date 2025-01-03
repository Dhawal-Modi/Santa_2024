import math
from typing import List
from metric import *

def greedy_reorder(words: List[str], perplexity_calculator, debug: bool = False) -> List[str]:
    remaining_words = words[:]
    reordered = []

    while remaining_words:
        best_word = None
        best_ppl = math.inf

        for w in remaining_words:
            candidate_sequence = reordered + [w]
            candiate_text = " ".join(candidate_sequence)

            ppl = perplexity_calculator.get_perplexity(candiate_text)

            if ppl < best_ppl:
                best_ppl = ppl
                best_word = w

        reordered.append(best_word)
        remaining_words.remove(best_word)

        if debug:
            print(f"Chosen word: {best_word}")
            print(f"Current partial sequence: {' '.join(reordered)}")
            print(f"Partial perplexity: {best_ppl:.3f}\n")

    return reordered

if __name__ == "__main__":
    scorer = PerplexityCalculator(model_path="google/gemma-2-9b")

    scrambled_words = "advent chimney elf family fireplace gingerbread mistletoe ornament reindeer scrooge"

    reordered_words = greedy_reorder(scrambled_words, scorer, debug=True)

    print("Greedy Reordered Sequence:")
    print(" ".join(reordered_words))