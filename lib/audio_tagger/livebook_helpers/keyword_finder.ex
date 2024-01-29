defmodule AudioTagger.LivebookHelpers.KeywordFinder do
  @moduledoc """
  Provides helper functions for using the `KeywordFinder` module in Livebook.

  Example output:
  ```
    [
      %{label: "known coronary artery disease", score: 0.35812485218048096},
      %{label: "old man", score: 0.13899555802345276},
      %{label: "comes", score: 0.12400370836257935},
      %{label: "visit today", score: 0.10444625467061996},
      %{label: "unstable angina", score: 0.09210294485092163}
    ]
  ```
  """
  alias AudioTagger.KeywordFinder

  def run(text) do
    phrases = extract_phrases(text)
    classify_text(text, phrases)
  end

  def extract_phrases(text) do
    serving = KeywordFinder.token_classification_serving()
    output = Nx.Serving.run(serving, text)
    entities = output.entities

    KeywordFinder.cleanup_phrases(entities)
  end

  def classify_text(text, labels) do
    serving = KeywordFinder.zero_shot_classification_serving(labels)
    output = Nx.Serving.run(serving, text)

    output.predictions
  end
end
