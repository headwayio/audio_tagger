defmodule AudioTagger.LivebookHelpers.SemanticSearch do
  @moduledoc """
  Provides a utility function for interacting with the `SemanticSearch` module from within Livebook.
  """

  alias AudioTagger.Classifier.SemanticSearch
  alias AudioTagger.Structs.SemanticSearchConfiguration
  alias AudioTagger.Utilities

  def tag(transcription_df, labels_df, label_vectors_path) do
    time_prep_start = System.monotonic_time()
    {model_info, tokenizer} = SemanticSearch.prepare_model()
    # labels = AudioTagger.Classifier.to_list_of_label_descriptions(labels_df)
    label_embeddings = SemanticSearch.load_label_vectors(label_vectors_path)
    Utilities.output_elapsed("Prepared model and loaded label embeddings", time_prep_start)

    IO.puts("Calculating similarity of transcribed text to labels in vector space")
    time_search_start = System.monotonic_time()

    input = %SemanticSearchConfiguration{
      model_info: model_info,
      tokenizer: tokenizer,
      labels_df: labels_df,
      label_embeddings: label_embeddings
    }

    tags =
      transcription_df
      |> Explorer.DataFrame.pull("text")
      |> Explorer.Series.downcase()
      |> Explorer.Series.transform(fn element ->
        SemanticSearch.tag_one(input, element)
      end)

    Utilities.output_elapsed(
      "Finished search for matching vectors for transcribed text",
      time_search_start
    )

    Explorer.DataFrame.put(transcription_df, "tags", tags)
  end
end
