defmodule AudioTagger.Vectors do
  @moduledoc """
  Precalculates vector embeddings for a list of strings.

  A future improvement may be to manage these vectors using HNSWLib:
   https://github.com/elixir-nx/hnswlib
  """
  alias AudioTagger.Utilities

  @doc "Creates vector embeddings for a given list of strings and stores those in a temp file for later use."
  def precalculate(input_filename) do
    output_file = output_filename(input_filename)

    if File.exists?(output_file) do
      IO.puts("Found pre-calculated vector embeddings. Skipping embedding.")
    else
      input_filename
      |> prepare_csv!()
      |> precalculate_label_vectors(output_file)

      IO.inspect(output_file, label: "Wrote vector embeddings")
    end
  end

  def embed(labels) do
    {model_info, tokenizer} = AudioTagger.Classifier.SemanticSearch.prepare_model()

    embed_with_model(model_info, tokenizer, labels)
  end

  def embed_with_model(model_info, tokenizer, labels) do
    label_inputs = Bumblebee.apply_tokenizer(tokenizer, labels)

    Axon.predict(model_info.model, model_info.params, label_inputs, compiler: EXLA)
  end

  def serving(model_name \\ "sentence-transformers/all-MiniLM-L6-v2") do
    {:ok, model_info} = Bumblebee.load_model({:hf, model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_name})

    Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer)
  end

  defp precalculate_label_vectors(labels_series, path) do
    time_label_start = System.monotonic_time()

    IO.puts("Creating vector embeddings for #{Explorer.Series.count(labels_series)} labels")

    label_embeddings =
      labels_series
      |> Explorer.Series.to_list()
      |> embed()

    Utilities.output_elapsed("Prepared label vector embeddings", time_label_start)

    iodata = Nx.serialize(label_embeddings)
    File.write(path, iodata)
  end

  # Builds a file path adjacent to the input filename with a different extension.
  defp output_filename(input_filename) do
    directory = Path.dirname(input_filename)
    filename_without_extension = Path.basename(input_filename, ".csv")

    Path.join(directory, "#{filename_without_extension}.bin")
  end

  # Loads a CSV file into an Explorer.DataFrame and prepares it for processing.
  defp prepare_csv!(filename) do
    filename
    |> Explorer.DataFrame.from_csv!(
      dtypes: [
        {"code", :string},
        {"long_description", :string}
      ]
    )
    |> Explorer.DataFrame.pull("long_description")
  end
end
