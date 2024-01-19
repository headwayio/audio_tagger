defmodule AudioTagger.Classifier.SemanticSearch do
  @moduledoc """
  Create embeddings for the labels and the portions of the transcribed text and then calculate cosine similarity to
  attempt to find the label that most closely matches a given portion of text.

  The number of the labels provided has an impact on performance, but this completes more quickly than the process in
  the TextClassification module (each is run with 14 text entries):
  - Local CPU (10 labels): 1.2s
  - Local CPU (1000 labels): 1.4s
  - Local CPU (10,000 labels): 16.0s
  - Local CPU (~73,000 labels): 113.2s (~7.5s for searching, the remainder for building the label vector embeddings)

  This is heavily based on the example given by Adrian Philipp (@adri) in https://github.com/elixir-nx/bumblebee/issues/100#issuecomment-1345563122
  """

  alias AudioTagger.Structs.SemanticSearchConfiguration
  alias AudioTagger.Structs.TagResult

  # Name of model to use for transforming text to vectors
  @model_name "sentence-transformers/all-MiniLM-L6-v2"
  # Number of matches to return
  @default_k 5
  # Minimum similarity score for returned matches
  @default_similarity_threshold 0.7

  @doc "Receives a configuration struct and a portion of text to search for within the list of vectors."
  def tag_one(%SemanticSearchConfiguration{labels_df: labels_df} = input, text) do
    labels = AudioTagger.Classifier.to_list_of_label_descriptions(labels_df)

    search_for_similar_codes(input, text)
    |> Enum.map(fn {index, score} ->
      {match_code, match_label} = find_label_for_index(index, labels, labels_df)

      %TagResult{code: match_code, label: match_label, score: score}
    end)
    |> Enum.filter(&(String.length(&1.label) > 0))
  end

  defp search_for_similar_codes(%SemanticSearchConfiguration{} = input, text, opts \\ []) do
    %{model_info: model_info, tokenizer: tokenizer, label_embeddings: label_embeddings} = input
    k = Keyword.get(opts, :num_results, @default_k)
    similarity_threshold = Keyword.get(opts, :similarity_threshold, @default_similarity_threshold)

    search_input = Bumblebee.apply_tokenizer(tokenizer, [text])

    search_embedding =
      Axon.predict(model_info.model, model_info.params, search_input, compiler: EXLA)

    similarities =
      Bumblebee.Utils.Nx.cosine_similarity(
        search_embedding.pooled_state,
        label_embeddings.pooled_state
      )

    # == Postprocessing ==

    # -- Find the top @k similar matches
    # values = [0.4, 0.5, 0.32, 0.7, 0.9]
    # indices_of_most_similar = [1200, 4500, 100, 340, 10]
    {values, indices_of_most_similar} = Nx.top_k(similarities, k: k)

    List.zip([
      Nx.to_flat_list(indices_of_most_similar),
      Nx.to_flat_list(values)
    ])
    # -- Remove matches that don't exceed a given threshold
    |> Enum.filter(fn {_index, score} -> score >= similarity_threshold end)

    # TODO: Potential improvement:
    # - If the highest code is distant from the next code, only return the highest code (e.g. [0.95, 0.7, 0.72, 0.73, 0.71] => [0.95])
    # - Or, if a number of top codes are close in value, return them all (e.g. [0.93, 0.71, 0.92, 0.80, 0.91] => [0.93, 0.92, 0.91])
  end

  defp find_label_for_index(index, labels, labels_df) do
    match_label = Enum.at(labels, index)

    match_code =
      AudioTagger.Classifier.code_for_label(labels_df, match_label)

    {match_code, match_label}
  end

  @doc "Loads a serialized binary file of vector embeddings from disk."
  def load_label_vectors(path) do
    {:ok, binary} = File.read(path)

    Nx.deserialize(binary)
  end

  @doc "Loads the model used for creating vector embeddings."
  def prepare_model do
    {:ok, model_info} = Bumblebee.load_model({:hf, @model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model_name})

    {model_info, tokenizer}
  end
end
