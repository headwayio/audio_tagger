defmodule AudioTagger.Transcriber do
  @model_name "openai/whisper-medium"
  # @model_name "openai/whisper-large" ~= 6 GB

  def transcribe_audio(featurizer, audio_file, num_channels) do
    serving = prepare_serving(featurizer)

    audio =
      audio_file
      |> Nx.from_binary(:f32)
      |> Nx.reshape({:auto, num_channels})
      |> Nx.mean(axes: [1])

    serving
    |> Nx.Serving.run(audio)
    |> Stream.map(fn chunk ->
      # TODO: We may be able to move this into `client_postprocessing/2
      [start_mark, end_mark] =
        for seconds <- [chunk.start_timestamp_seconds, chunk.end_timestamp_seconds] do
          seconds |> round() |> Time.from_seconds_after_midnight() |> Time.to_string()
        end

      %{start_mark: start_mark, end_mark: end_mark, text: chunk.text}
    end)

    # |> Explorer.DataFrame.new()
  end

  def prepare_featurizer do
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, @model_name})

    featurizer
  end

  def prepare_serving(featurizer) do
    {:ok, model_info} = Bumblebee.load_model({:hf, @model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model_name})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, @model_name})
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
