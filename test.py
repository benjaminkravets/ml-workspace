import pandas as pd

def calculate_weighted_average_cv(category_float_pairs):
  """Calculates the weighted average coefficient of variation (weighted average CV) for a group of category and float pairs.

  Args:
    category_float_pairs: A list of category and float pairs, where each pair is a tuple of (category, float value).

  Returns:
    The weighted average CV.
  """

  # Create a DataFrame from the list of pairs.
  df = pd.DataFrame(category_float_pairs, columns=["category", "float"])

  # Calculate the mean float value for each category.
  mean_float_per_category = df.groupby("category")["float"].mean()

  # Calculate the standard deviation of the float values for each category.
  std_float_per_category = df.groupby("category")["float"].std()

  # Calculate the coefficient of variation (CV) for each category.
  cv_per_category = std_float_per_category / mean_float_per_category

  # Calculate the number of data points in each category.
  count_per_category = df.groupby("category")["float"].count()

  # Calculate the weighted average CV.
  weighted_average_cv = (cv_per_category * count_per_category).sum() / count_per_category.sum()

  return weighted_average_cv

# Example usage:

category_float_pairs = [("A", 1.3), ("B", 2.4), ("B", 2.4), ("C", 3.6), ("C", 3.6)]

weighted_average_cv = calculate_weighted_average_cv(category_float_pairs)

print(weighted_average_cv)
