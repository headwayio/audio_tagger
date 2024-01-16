defmodule AudioTagger.Vectors do
  @moduledoc """
  Precalculates vector embeddings for a list of strings.
  """

  @doc "Creates vector embeddings for a given list of strings and stores those in a temp file for later use."
  def precalculate(input_filename) do
    output_file = output_filename(input_filename)

    if File.exists?(output_file) do
      IO.puts("Found pre-calculated vector embeddings. Skipping embedding.")
    else
      input_filename
      |> prepare_csv!()
      |> AudioTagger.Classifier.SemanticSearch.precalculate_label_vectors(output_file)

      IO.inspect(output_file, label: "Wrote vector embeddings")
    end
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
        {"CODE", :string},
        {"DIAGNOSIS CODE", :string},
        {"LONG DESCRIPTION", :string},
        {"SHORT DESCRIPTION", :string}
      ]
    )
    |> Explorer.DataFrame.pull("LONG DESCRIPTION")
  end
end
