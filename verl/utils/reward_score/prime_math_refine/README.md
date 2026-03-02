# prime_math_refine

Refined version of `prime_math` with the following improvements:

## Key Features

1. **Only extracts answer from `\boxed{}`**: Prevents extracting numbers from calculation process (e.g., `6^{21}` → "21")
2. **Strict matching for integers**: No tolerance for integer comparisons
3. **Flexible handling for mathematical expressions**: Uses sympy for mathematical equivalence checking
4. **No tolerance for large integers**: Strict integer matching without floating-point tolerance

## Usage

```python
from verl.utils.reward_score.prime_math_refine import compute_score

# Example 1: Integer comparison (strict, no tolerance)
model_output = "The answer is \\boxed{21}"
ground_truth = "21"
is_correct, format_correct, extracted = compute_score(model_output, ground_truth)
# Returns: (True, True, "21")

# Example 2: Integer mismatch (strict, no tolerance)
model_output = "The answer is \\boxed{25}"
ground_truth = "21"
is_correct, format_correct, extracted = compute_score(model_output, ground_truth)
# Returns: (False, True, "25") - correctly identifies mismatch

# Example 3: Mathematical expression (flexible)
model_output = "The answer is \\boxed{\\frac{1}{2}}"
ground_truth = "0.5"
is_correct, format_correct, extracted = compute_score(model_output, ground_truth)
# Returns: (True, True, "\\frac{1}{2}") - recognizes mathematical equivalence

# Example 4: No \\boxed{} found
model_output = "The answer is 21"
ground_truth = "21"
is_correct, format_correct, extracted = compute_score(model_output, ground_truth)
# Returns: (False, False, None) - no \\boxed{} found
```

## Differences from prime_math

| Feature | prime_math | prime_math_refine |
|---------|------------|-------------------|
| Answer extraction | Multiple formats (answer:, \\boxed{}, etc.) | Only `\boxed{}` |
| Integer matching | May use tolerance | Strict matching (no tolerance) |
| Mathematical expressions | Flexible (sympy) | Flexible (sympy) |
| Large integer tolerance | May use tolerance | No tolerance |

## Integration

To use `prime_math_refine` in your reward scoring, update `verl/utils/reward_score/__init__.py`:

```python
from . import prime_math_refine
res = prime_math_refine.compute_score(solution_str, ground_truth)
```

Or use it directly in your code:

```python
from verl.utils.reward_score.prime_math_refine import compute_score
result = compute_score(model_output, ground_truth)
```

