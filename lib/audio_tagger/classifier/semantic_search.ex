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
  alias AudioTagger.Classifier.TagResult

  # Number of matches to return
  @k 5
  # Minimum similarity score for returned matches
  @similarity_threshold 0.7

  def precalculate_label_vectors(labels_series, path) do
    time_label_start = System.monotonic_time()

    label_embeddings =
      labels_series
      |> Explorer.Series.to_list()
      |> embed_label_vectors()

    output_elapsed("Prepared label vector embeddings", time_label_start)

    iodata = Nx.serialize(label_embeddings)
    File.write(path, iodata)
  end

  def tag_one(model_tuple, labels_df, label_embeddings, element) do
    {model_info, tokenizer} = model_tuple
    labels = AudioTagger.Tagger.to_list_of_label_descriptions(labels_df)

    search_for_similar_codes(model_info, tokenizer, label_embeddings, element)
    |> Enum.map(fn {index, score} ->
      {match_code, match_label} = find_label_for_index(index, labels, labels_df)

      %TagResult{code: match_code, label: match_label, score: score}
    end)
  end

  def tag(transcription_df, labels_df, label_vectors_path) do
    time_prep_start = System.monotonic_time()
    {model_info, tokenizer} = prepare_model()
    # labels = AudioTagger.Tagger.to_list_of_label_descriptions(labels_df)
    label_embeddings = load_label_vectors(label_vectors_path)
    output_elapsed("Prepared model and loaded label embeddings", time_prep_start)

    IO.puts("Calculating similarity of transcribed text to labels in vector space")
    time_search_start = System.monotonic_time()

    tags =
      transcription_df
      |> Explorer.DataFrame.pull("text")
      |> Explorer.Series.downcase()
      |> Explorer.Series.transform(fn element ->
        tag_one({model_info, tokenizer}, labels_df, label_embeddings, element)
      end)

    output_elapsed("Finished search for matching vectors for transcribed text", time_search_start)

    Explorer.DataFrame.put(transcription_df, "tags", tags)
  end

  defp embed_label_vectors(labels) do
    {model_info, tokenizer} = prepare_model()
    IO.puts("Creating vector embeddings for #{Enum.count(labels)} labels")

    label_inputs = Bumblebee.apply_tokenizer(tokenizer, labels)

    Axon.predict(model_info.model, model_info.params, label_inputs, compiler: EXLA)
  end

  defp search_for_similar_codes(model_info, tokenizer, label_embeddings, element) do
    search_input = Bumblebee.apply_tokenizer(tokenizer, [element])

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
    {values, indices_of_most_similar} = Nx.top_k(similarities, k: @k)

    List.zip([
      Nx.to_flat_list(indices_of_most_similar),
      Nx.to_flat_list(values)
    ])
    # -- Remove matches that don't exceed a given threshold
    |> Enum.filter(fn {_index, score} -> score >= @similarity_threshold end)

    # TODO: Potential improvement:
    # - If the highest code is distant from the next code, only return the highest code (e.g. [0.95, 0.7, 0.72, 0.73, 0.71] => [0.95])
    # - Or, if a number of top codes are close in value, return them all (e.g. [0.93, 0.71, 0.92, 0.80, 0.91] => [0.93, 0.92, 0.91])
  end

  defp find_label_for_index(index, labels, labels_df) do
    match_label = Enum.at(labels, index)

    match_code =
      AudioTagger.Tagger.code_for_label(labels_df, match_label)

    # |> IO.inspect()

    {match_code, match_label}
  end

  def load_label_vectors(path) do
    {:ok, binary} = File.read(path)

    Nx.deserialize(binary)
  end

  def prepare_model do
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    {:ok, model_info} = Bumblebee.load_model({:hf, model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_name})

    {model_info, tokenizer}
  end

  defp output_elapsed(label, time_start) do
    time_end = System.monotonic_time()

    IO.puts(
      "#{label}. Took #{System.convert_time_unit(time_end - time_start, :native, :millisecond)}ms"
    )
  end
end
