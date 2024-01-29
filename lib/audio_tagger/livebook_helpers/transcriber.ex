defmodule AudioTagger.LivebookHelpers.Transcriber do
  @moduledoc """
  This module holds functions for interaction with audio transcription from within a Livebook session.
  """

  @default_model_name "openai/whisper-medium"

  def transcribe_audio(featurizer, audio_file, opts \\ []) do
    num_channels = Keyword.get(opts, :num_channels, 2)
    model_name = Keyword.get(opts, :model_name, @default_model_name)

    serving = AudioTagger.Transcriber.serving_with_featurizer(featurizer, model_name)

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
  end

  def prepare_featurizer(opts \\ []) do
    model_name = Keyword.get(opts, :model_name, @default_model_name)
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, model_name})

    featurizer
  end
end
