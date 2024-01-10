defmodule Mix.Tasks.Embedding do
  @moduledoc "Generate embeddings from ICD-9 codes"

  use Mix.Task

  @shortdoc "Generates icd10_vector_tensors.bin in system tmp"
  def run(args) do
    Mix.Task.run("app.start")

    csv = Enum.at(args, 0)

    csv_filename = Path.basename(csv, ".csv")

    tmpfile = Path.join(System.tmp_dir(), "#{csv_filename}.bin")

    if File.exists?(tmpfile) do
      IO.puts("Found pre-calculated ICD-9 vector embeddings. Skipping embedding.")
    else
      csv
      |> Explorer.DataFrame.from_csv!(
        dtypes: [
          {"DIAGNOSIS CODE", :string},
          {"LONG DESCRIPTION", :string},
          {"SHORT DESCRIPTION", :string}
        ]
      )
      |> Explorer.DataFrame.pull("SHORT DESCRIPTION")
      |> AudioTagger.Classifier.SemanticSearch.precalculate_label_vectors(tmpfile)

      IO.inspect(tmpfile, label: "Wrote vector embeddings")
    end
  end
end
