defmodule AudioTagger.Transcriber do
  @moduledoc """
  Contains functions to prepare a speech-to-text model for use in transcribing text.
  """

  @default_model_name "openai/whisper-medium"

  @doc "Creates an Nx.Serving to perform speech-to-text tasks."
  def serving(model_name \\ @default_model_name) do
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, model_name})

    serving_with_featurizer(featurizer, model_name)
  end

  @doc "Creates an Nx.Serving to perform speech-to-text tasks, using the passed featurizer. This is helpful for direct use from Livebook where the featurizer is needed to define the Kino audio input."
  def serving_with_featurizer(featurizer, model_name \\ @default_model_name) do
    {:ok, model_info} = Bumblebee.load_model({:hf, model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_name})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, model_name})
    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 100)

    # Docs: https://hexdocs.pm/bumblebee/Bumblebee.Audio.html#speech_to_text_whisper/5
    Bumblebee.Audio.speech_to_text_whisper(
      model_info,
      featurizer,
      tokenizer,
      generation_config,
      compile: [batch_size: 4],
      chunk_num_seconds: 30,
      # context_num_seconds: 5, # Defaults to 1/6 of :chunk_num_seconds
      timestamps: :segments,
      stream: true,
      defn_options: [compiler: EXLA]
    )
  end
end
