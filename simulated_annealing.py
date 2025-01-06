import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from metric import PerplexityCalculator
import random
import math

class TextReorderer:
    def __init__(self, model_path: str = "google/gemma-2-9b"):
        """Initialize the reorderer with a perplexity calculator."""
        self.calculator = PerplexityCalculator(model_path=model_path)

    def _get_perplexity(self, text: str) -> float:
        """Calculate perplexity for a single text string."""
        return self.calculator.get_perplexity(text)

    def _swap_words(self, words: List[str], i: int, j: int) -> List[str]:
        """Swap two words in a list and return new list."""
        words = words.copy()
        words[i], words[j] = words[j], words[i]
        return words

    def _get_random_swap_indices(self, n: int, max_distance: Optional[int] = None) -> Tuple[int, int]:
        """Get random indices for swapping, optionally within a maximum distance."""
        if max_distance is None:
            i = random.randint(0, n-2)
            j = random.randint(i+1, n-1)
        else:
            i = random.randint(0, n-2)
            max_j = min(i + max_distance, n-1)
            j = random.randint(i+1, max_j)
        return i, j

    def reorder_with_simulated_annealing(
            self,
            text: str,
            initial_temp: float = 100.0,
            final_temp: float = 0.0,
            max_iterations: int = 5000,
            cooling_rate: float = 0.95,
            max_distance: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Reorder text using simulated annealing.

        Args:
            text: Space-separated string of words to reorder
            initial_temp: Starting temperature
            final_temp: Temperature at which to stop
            max_iterations: Maximum number of iterations
            cooling_rate: Rate at which temperature decreases
            max_distance: Maximum distance between words that can be swapped

        Returns:
            Tuple of (reordered text, final perplexity score)
        """
        # Split into words and get initial state
        words = text.split()
        current_text = " ".join(words)
        current_perplexity = self._get_perplexity(current_text)

        # Track best solution
        best_words = words.copy()
        best_perplexity = current_perplexity

        # Initialize temperature
        temp = initial_temp
        iteration = 0

        # Main simulated annealing loop
        while temp > final_temp and iteration < max_iterations:
            # Get random swap indices
            i, j = self._get_random_swap_indices(len(words), max_distance)

            # Create candidate solution
            candidate_words = self._swap_words(words, i, j)
            candidate_text = " ".join(candidate_words)
            candidate_perplexity = self._get_perplexity(candidate_text)

            if candidate_perplexity < best_perplexity:
                best_words = candidate_words.copy()
                #best_text =  " ".join(best_words)
                best_perplexity = candidate_perplexity
                #print('>%d %s = %.5f' % (iteration, best_text, best_perplexity))

            diff = candidate_perplexity - current_perplexity
            exponent = -diff / temp

            # Clamp exponent to avoid overflow/underflow
            # For example, restrict to [-700, 700]. Adjust these bounds to your needs.
            exponent = max(min(exponent, 700), -700)

            acceptance_criterion = math.exp(exponent)

            if diff < 0 or random.random() < acceptance_criterion:
                words = candidate_words
                current_perplexity = candidate_perplexity

            # Cool down
            temp = temp - ((initial_temp - final_temp) / max_iterations)
            # temp = initial_temp / math.log(iteration + 2)
            # temp *= cooling_rate
            iteration += 1

            # Periodic logging
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Temp: {temp:.9f}, Best Perplexity: {best_perplexity:.4f}")

        return " ".join(best_words), best_perplexity

    def reorder_dataset(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Reorder all texts in a dataset.

        Args:
            df: DataFrame with 'id' and 'text' columns
            **kwargs: Additional arguments passed to the chosen method

        Returns:
            DataFrame with reordered texts
        """
        results = []
        perplexities = []

        for _, row in df.iterrows():
            reordered_text, perplexity = self.reorder_with_simulated_annealing(row['text'], **kwargs)

            results.append({
                'id': row['id'],
                'text': reordered_text
            })
            perplexities.append(perplexity)

        result_df = pd.DataFrame(results)
        print(f"Average perplexity: {np.mean(perplexities):.2f}")
        return result_df

def create_submission(input_path: str, output_path: str, model_path: str = "google/gemma-2-9b", **kwargs) -> None:
    """
    Create a submission file from input data.

    Args:
        input_path: Path to input CSV
        output_path: Path to save submission
        model_path: Path to the language model
        **kwargs: Additional arguments for the chosen method
    """
    df = pd.read_csv(input_path)
    reorderer = TextReorderer(model_path=model_path)
    result_df = reorderer.reorder_dataset(df, **kwargs)
    result_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    # Create submission
    create_submission(input_path="data/textidone.csv",output_path="submission_id1.csv")