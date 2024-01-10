defmodule AudioTagger.Tagger do
  @moduledoc """
  Takes an Explorer dataframe containing transcribed text from an audio recording and runs each entry through a
  zero-shot text classification model to tag the audio.
  """

  require Explorer.DataFrame

  # def tag_audio(transcription_df, labels_df, classifier_type \\ :semantic_search) do
  #   case classifier_type do
  #     :semantic_search -> AudioTagger.Classifier.SemanticSearch.tag(transcription_df, labels_df)
  #     :text_classification -> AudioTagger.Classifier.SemanticSearch.tag(transcription_df, labels_df)
  #   end
  # end

  def code_for_label(labels_df, description) do
    Explorer.DataFrame.filter(labels_df, short_description == ^description)
    |> Explorer.DataFrame.pull("code")
    |> Explorer.Series.first()
  end

  def to_list_of_label_descriptions(labels_df) do
    labels_df
    |> Explorer.Series.to_list()
  end
end
