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

  def precalculate_label_vectors(labels_df, path) do
    time_label_start = System.monotonic_time()
    label_embeddings = embed_label_vectors(labels_df)
    output_elapsed("Prepared label vector embeddings", time_label_start)

    iodata = Nx.serialize(label_embeddings)
    File.write(path, iodata)
  end

  def tag(transcription_df, labels_df, label_vectors_path) do
    time_prep_start = System.monotonic_time()
    {model_info, tokenizer} = prepare_model()
    labels = AudioTagger.Tagger.to_list_of_label_descriptions(labels_df)
    label_embeddings = load_label_vectors(label_vectors_path)
    output_elapsed("Prepared model and loaded label embeddings", time_prep_start)

    IO.puts("Calculating similarity of transcribed text to labels in vector space")
    time_search_start = System.monotonic_time()

    tags =
      transcription_df
      |> Explorer.DataFrame.pull("text")
      |> Explorer.Series.downcase()
      |> Explorer.Series.transform(fn element ->
        match_index = search_for_similar_code(model_info, tokenizer, label_embeddings, element)
        {match_code, match_label} = find_label_for_index(match_index, labels, labels_df)

        "#{match_code}: #{match_label}"
      end)

    output_elapsed("Finished search for matching vectors for transcribed text", time_search_start)

    Explorer.DataFrame.put(transcription_df, "tags", tags)
  end

  defp embed_label_vectors(labels_df) do
    {model_info, tokenizer} = prepare_model()
    labels = AudioTagger.Tagger.to_list_of_label_descriptions(labels_df)
    IO.puts("Creating vector embeddings for #{Enum.count(labels)} labels")

    label_inputs = Bumblebee.apply_tokenizer(tokenizer, labels)

    Axon.predict(model_info.model, model_info.params, label_inputs, compiler: EXLA)
  end

  defp search_for_similar_code(model_info, tokenizer, label_embeddings, element) do
    search_input = Bumblebee.apply_tokenizer(tokenizer, [element])

    search_embedding =
      Axon.predict(model_info.model, model_info.params, search_input, compiler: EXLA)

    Bumblebee.Utils.Nx.cosine_similarity(
      search_embedding.pooled_state,
      label_embeddings.pooled_state
    )
    |> Nx.argmax()
    |> Nx.to_number()
    |> dbg()
  end

  defp find_label_for_index(index, labels, labels_df) do
    match_label = Enum.at(labels, index)

    match_code =
      AudioTagger.Tagger.code_for_label(labels_df, match_label)
      |> IO.inspect()

    {match_code, match_label}
  end

  defp load_label_vectors(path) do
    {:ok, binary} = File.read(path)

    Nx.deserialize(binary)
  end

  defp prepare_model do
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
