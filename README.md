# AudioTagger

Take an audio recording, transcribe it to text, and then tag portions of audio based on a list of terms.

The initial use case is to find mentions of medical procedures to label the audio with procedure codes.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `audio_tagger` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:audio_tagger, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/audio_tagger>.

## Performance

Audio transcription runs rather quickly. An audio file of around a minute can be transcribed in five to six seconds.

On the other hand, text-classification with a large set of labels can be quite slow:
- Within Livebook, running a single text input against the entire ICD-10 code list with around 73,000 labels doesn't complete.
- Running a single text input against 1,000 labels takes around two minutes (139.4 seconds in one case).
- Running the entire audio transcription of 14 text entries with 1,000 labels took 1793.8s (almost a half hour).

## Next Steps

1. Run with a GPU on Hugging Faces space
2. Possibly consider other models that may be more performant or smaller (for example, https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)

We've discussed processing the text from the audio transcription before classification (by removing stop words and
lemmatizing / finding the root of individual words), but with our use of a text-classification model this may not be
valuable.
