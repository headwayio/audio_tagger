defmodule AudioTagger.Tagger do
  @moduledoc """
  Takes an Explorer dataframe containing transcribed text from an audio recording and runs each entry through a
  zero-shot text classification model to tag the audio.
  """

  require Explorer.DataFrame

  @doc """
  Requires:
    - a transcription_df as returned by AudioTagger.Transcriber.transcribe_audio
    - and a labels_df that contains "code" and "short_description" columns
  """
  def tag_audio(transcription_df, labels_df) do
    labels = prepare_labels(labels_df)
    serving = prepare_serving(labels)

    tags =
      transcription_df
      |> Explorer.DataFrame.pull("text")
      |> Explorer.Series.downcase()
      |> Explorer.Series.transform(fn element ->
        classification = classify_text(serving, element)

        {:ok, {description, _}} =
          classification
          |> Enum.fetch(0)

        code_for_label(labels_df, description)
        |> IO.inspect(label: "Detected code for #{element}")
      end)

    transcription_df
    |> Explorer.DataFrame.put("tags", tags)
  end

  def classify_text(serving, text) do
    output = Nx.Serving.run(serving, text)

    output.predictions
    |> Enum.map(&{&1.label, &1.score})
  end

  def code_for_label(labels_df, description) do
    Explorer.DataFrame.filter(labels_df, short_description == ^description)
    |> Explorer.DataFrame.pull("code")
    |> Explorer.Series.first()
  end

  def prepare_labels(labels_df) do
    labels_df
    |> Explorer.DataFrame.pull("short_description")
    |> Explorer.Series.to_list()
  end

  def prepare_serving(labels) do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})

    Bumblebee.Text.zero_shot_classification(model_info, tokenizer, labels,
      compile: [batch_size: 1, sequence_length: 100],
      defn_options: [compiler: EXLA]
    )
  end
end
