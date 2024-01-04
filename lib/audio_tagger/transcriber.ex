defmodule AudioTagger.Transcriber do
  def transcribe_audio(featurizer, audio_file, num_channels) do
    serving = prepare_serving(featurizer)

    audio =
      audio_file
      |> Nx.from_binary(:f32)
      |> Nx.reshape({:auto, num_channels})
      |> Nx.mean(axes: [1])

    Nx.Serving.run(serving, audio)
    |> Enum.reduce([], fn chunk, acc ->
      [start_mark, end_mark] =
        for seconds <- [chunk.start_timestamp_seconds, chunk.end_timestamp_seconds] do
          seconds |> round() |> Time.from_seconds_after_midnight() |> Time.to_string()
        end

      [%{start_mark: start_mark, end_mark: end_mark, text: chunk.text}] ++ acc
    end)
    |> Enum.reverse()
    |> Explorer.DataFrame.new()
  end

  def prepare_featurizer do
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})

    featurizer
  end

  defp prepare_serving(featurizer) do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})
    # {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/whisper-tiny"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai/whisper-tiny"})
    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 100)

    Bumblebee.Audio.speech_to_text_whisper(
      model_info,
      featurizer,
      tokenizer,
      generation_config,
      compile: [batch_size: 4],
      chunk_num_seconds: 30,
      timestamps: :segments,
      stream: true,
      defn_options: [compiler: EXLA]
    )
  end
end
