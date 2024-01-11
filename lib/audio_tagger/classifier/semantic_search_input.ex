defmodule AudioTagger.Classifier.SemanticSearchInput do
  @enforce_keys [:model_info, :tokenizer, :labels_df, :label_embeddings, ]
  defstruct model_info: nil, tokenizer: nil, labels_df: nil, label_embeddings: nil, opts: []
  # @type t :: %__MODULE__{label: String.t(), code: String.t(), score: float()}
end
