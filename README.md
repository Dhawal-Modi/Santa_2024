# The Challenge

- Just like those mischievous elves who mixed up the ornaments on Santa's Christmas tree, someone has scrambled the words in classic Christmas  tales! Your task, dear Kagglers, is to put those words back in order, minimizing the perplexity of each passage. The lower the perplexity, the more sense the story makes!

# The Data

- Rudolph has provided a sleigh-full of jumbled text passages, each one a beloved Christmas story in disarray. He's even included a special metric to guide you, a measure of how puzzled a reader would be by the mixed-up words.

# The Goal

- Use your coding magic and linguistic skills to rearrange the words, making the stories flow smoothly and beautifully once more. The Kagglers who achieve the lowest perplexity scores will win Rudolph's heart and earn a place on the leaderboard!

- So join Rudolph and his friends in this festive challenge! Untangle the words, spread Christmas cheer, and help make this a holiday to remember!

# Evaluation

- Submissions to this competition are evaluated by their Average Perplexity, with lower scores ranked better.

- Submissions comprise texts that are permutations of words given in the sample_submission.csv file. We compute the perplexity of each submitted sequence of words with respect to the Gemma 2 9B model. Your final score is the average perplexity across all sequences.

- You may view the evaluation metric in this python file: metric.py.

# Submission File

- For each id, the submission file should contain a sequence of words in the text field. Each sequence must be attainable by a permutation of the corresponding "base" sequence of words in the sample_submission.csv file. For example, if the base sequence is a b c, then the submitted sequence could be a b c, a c b, b a c, b c a, c a b, or c b a, but the submitted sequence could not be b a or a a b or a b c b. Such sequences are invalid and will fail scoring.

- The submission.csv file should contain a header and have the following format:

``` id,text
0,advent chimney elf family ...
1,fireplace gingerbread mistletoe ...
2,yuletide decorations gifts ...
...
```