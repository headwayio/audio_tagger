defmodule AudioTagger.Vectors do
  @moduledoc """
  Precalculates vector embeddings for a list of strings.
  """

  @shortdoc "Loads a CSV file into an Explorer.DataFrame and prepares it for processing."
  def prepare_csv!(filename) do
    filename
      |> Explorer.DataFrame.from_csv!(
        dtypes: [
          {"DIAGNOSIS CODE", :string},
          {"LONG DESCRIPTION", :string},
          {"SHORT DESCRIPTION", :string}
        ]
      )
      |> Explorer.DataFrame.pull("SHORT DESCRIPTION")
  end

  @shortdoc "Creates vector embeddings for a given list of strings and stores those in a temp file for later use."
  def precalculate(input_filename) do
    csv_filename = Path.basename(input_filename, ".csv")
    output_file = Path.join(System.tmp_dir(), "#{csv_filename}.bin")

    if File.exists?(output_file) do
      IO.puts("Found pre-calculated vector embeddings. Skipping embedding.")
    else
      input_filename
      |> prepare_csv!()
      |> AudioTagger.Classifier.SemanticSearch.precalculate_label_vectors(output_file)

      IO.inspect(output_file, label: "Wrote vector embeddings")
    end
  end
end
