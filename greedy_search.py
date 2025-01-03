import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from metric import PerplexityCalculator

class BaselineReorderer:
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
    
    def reorder_single_text(self, text: str, max_iterations: int = 100) -> Tuple[str, float]:
        """
        Reorder a single text using greedy local search.
        
        Args:
            text: Space-separated string of words to reorder
            max_iterations: Maximum number of swap attempts
            
        Returns:
            Tuple of (reordered text, final perplexity score)
        """
        # Split into words and get initial perplexity
        words = text.split()
        current_text = " ".join(words)
        current_perplexity = self._get_perplexity(current_text)
        
        # Track improvements
        iterations_without_improvement = 0
        
        for _ in tqdm(range(max_iterations)):
            improved = False
            
            # Try swapping each adjacent pair
            for i in range(len(words) - 1):
                # Create candidate swap
                candidate_words = self._swap_words(words, i, i + 1)
                candidate_text = " ".join(candidate_words)
                candidate_perplexity = self._get_perplexity(candidate_text)
                
                # If better, accept the swap
                if candidate_perplexity < current_perplexity:
                    words = candidate_words
                    current_perplexity = candidate_perplexity
                    improved = True
                    iterations_without_improvement = 0
                    break
            
            # If no improvement found, increment counter
            if not improved:
                iterations_without_improvement += 1
            
            # Early stopping if no recent improvements
            if iterations_without_improvement >= 10:
                break
        
        return " ".join(words), current_perplexity
    
    def reorder_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder all texts in a dataset.
        
        Args:
            df: DataFrame with 'id' and 'text' columns
            
        Returns:
            DataFrame with reordered texts
        """
        results = []
        perplexities = []
        
        for _, row in df.iterrows():
            reordered_text, perplexity = self.reorder_single_text(row['text'])
            results.append({
                'id': row['id'],
                'text': reordered_text
            })
            perplexities.append(perplexity)
        
        result_df = pd.DataFrame(results)
        print(f"Average perplexity: {np.mean(perplexities):.2f}")
        return result_df

def create_submission(
    input_path: str,
    output_path: str,
    model_path: str = "google/gemma-2-9b"
) -> None:
    """
    Create a submission file from input data.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save submission
        model_path: Path to the language model
    """
    # Read input data
    df = pd.read_csv(input_path)
    
    # Initialize reorderer
    reorderer = BaselineReorderer(model_path=model_path)
    
    # Process all texts
    result_df = reorderer.reorder_dataset(df)
    
    # Save submission
    result_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    # Create submission
    create_submission(input_path="data/spp_submission.csv",output_path="submission.csv")