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

  @model_name "facebook/bart-large-mnli"

  @doc """
  Requires:
    - a transcription_df as returned by AudioTagger.Transcriber.transcribe_audio
    - and a labels_df that contains "code" and "long_description" columns
  """
  def tag(transcription_df, labels_df) do
    labels = AudioTagger.Classifier.to_list_of_label_descriptions(labels_df)
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

        code =
          AudioTagger.Classifier.code_for_label(labels_df, description)
          |> IO.inspect(label: "Detected code for #{element}")

        "#{code}: #{description}"
      end)

    transcription_df
    |> Explorer.DataFrame.put("tags", tags)
  end

  defp classify_text(serving, text) do
    output = Nx.Serving.run(serving, text)

    output.predictions
    |> Enum.map(&{&1.label, &1.score})
  end

  @doc "Loads the model and prepares the serving for use."
  def prepare_serving(labels) do
    {:ok, model_info} = Bumblebee.load_model({:hf, @model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model_name})

    Bumblebee.Text.zero_shot_classification(model_info, tokenizer, labels,
      compile: [batch_size: 1, sequence_length: 100],
      defn_options: [compiler: EXLA]
    )
  end
end
