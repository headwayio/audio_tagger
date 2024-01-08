defmodule AudioTagger.Classifier.SemanticSearch do
  @moduledoc """
  Create embeddings for the labels and the portions of the transcribed text and then calculate cosine similarity to
  attempt to find the label that most closely matches a given portion of text.

  The number of the labels provided has an impact on performance, but this completes more quickly than the process in
  the TextClassification module (each is run with 14 text entries):
  - Local CPU (10 labels): 1.2s
  - Local CPU (1000 labels): 1.4s
  - Local CPU (10,000 labels): 16.0s
  - Local CPU (~73,000 labels): 113.2s
  
  This is heavily based on the example given by Adrian Philipp (@adri) in https://github.com/elixir-nx/bumblebee/issues/100#issuecomment-1345563122
  """

  def tag(transcription_df, labels_df) do
    time_prep_start = System.monotonic_time()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    {:ok, model_info} = Bumblebee.load_model({:hf, model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_name})

    labels = AudioTagger.Tagger.prepare_labels(labels_df)
    label_inputs = Bumblebee.apply_tokenizer(tokenizer, labels)
    output_elapsed("Prepared model", time_prep_start)

    time_label_start = System.monotonic_time()
    IO.puts("Creating vector embeddings for #{Enum.count(labels)} labels")
    label_embeddings =
      Axon.predict(model_info.model, model_info.params, label_inputs, compiler: EXLA)
    output_elapsed("Prepared label vector embeddings", time_label_start)

    time_search_start = System.monotonic_time()
    IO.puts("Calculating similarity of transcribed text to labels in vector space")
    tags =
      transcription_df
      |> Explorer.DataFrame.pull("text")
      |> Explorer.Series.downcase()
      |> Explorer.Series.transform(fn element ->
        search_input = Bumblebee.apply_tokenizer(tokenizer, [element])

        search_embedding =
          Axon.predict(model_info.model, model_info.params, search_input, compiler: EXLA)

        match_index =
          Bumblebee.Utils.Nx.cosine_similarity(
            search_embedding.pooled_state,
            label_embeddings.pooled_state
          )
          |> Nx.argmax()
          |> Nx.to_number()

        match_label = Enum.at(labels, match_index)

        match_code =
          AudioTagger.Tagger.code_for_label(labels_df, match_label)
          |> IO.inspect(label: "Dectected code for #{element}")

          "#{match_code}: #{match_label}"
      end)
    output_elapsed("Finished search for matching vectors for transcribed text", time_search_start)

    transcription_df
    |> Explorer.DataFrame.put("tags", tags)
  end

  defp output_elapsed(label, time_start) do
    time_end = System.monotonic_time()

    IO.puts("#{label}. Took #{System.convert_time_unit(time_end - time_start, :native, :millisecond)}ms")
  end
end
