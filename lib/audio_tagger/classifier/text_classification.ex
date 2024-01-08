defmodule AudioTagger.Classifier.TextClassification do
  @moduledoc """
  Run the portions of the transcribed text through the facebook/bart-large-mnli model and do zero-shot
  text-classification to attempt to tag each with a procedure code (based on the ICD-10 code list).
  
  The number of labels provided to the model has a direct impact on performance (each is run with 14 text entries):
  - Local CPU (1,000 labels): 1793.8s ~= 29+m
  - Local CPU (10 labels): 31.8s
  - Nvidia T4 on HF (1,000 labels): 246.3s
  - Nvidia A10G on HF (1,000 labels): 50s
  - Nvidia A10G on HF (2,500 labels): 130s
  """

  require Explorer.DataFrame

  @doc """
  Requires:
    - a transcription_df as returned by AudioTagger.Transcriber.transcribe_audio
    - and a labels_df that contains "code" and "short_description" columns
  """
  def tag(transcription_df, labels_df) do
    labels = AudioTagger.Tagger.prepare_labels(labels_df)
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

        code = AudioTagger.Tagger.code_for_label(labels_df, description)
        |> IO.inspect(label: "Detected code for #{element}")

        "#{code}: #{description}"
      end)

    transcription_df
    |> Explorer.DataFrame.put("tags", tags)
  end

  def classify_text(serving, text) do
    output = Nx.Serving.run(serving, text)

    output.predictions
    |> Enum.map(&{&1.label, &1.score})
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
