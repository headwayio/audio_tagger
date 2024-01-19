defmodule AudioTagger.Structs.SemanticSearchConfiguration do
  @moduledoc """
  Represents the model and label information used by the `SemanticSearch` model.

  - model_info: Model information from Bumblebee's `load_model`
  - tokenizer: Tokenizer from Bumblebee's `load_tokenizer`
  - labels_df: An Explorer.DataFrame containing the list of `long_description` and `code` pairs for looking up matched codes
  - label_embeddings: Vector embeddings of `long_description` text for searching against
  """
  @enforce_keys [:model_info, :tokenizer, :labels_df, :label_embeddings]
  defstruct model_info: nil, tokenizer: nil, labels_df: nil, label_embeddings: nil, opts: []
end
