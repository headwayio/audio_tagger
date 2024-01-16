defmodule Mix.Tasks.Embeddings do
  @moduledoc "Generate embeddings from downloaded ICD-9 code list"

  use Mix.Task

  @shortdoc "Generates icd10_vector_tensors.bin in system tmp"
  def run(args) do
    Mix.Task.run("app.start")

    if Enum.count(args) == 0 do
      # No arguments. Download and use ICD-9 code list.
      unless File.exists?(cached_download()) do
        AudioTagger.SampleData.get_icd9_code_list_csv()
      end

      AudioTagger.Vectors.precalculate(cached_download())
    else
      csv = Enum.at(args, 0)

      AudioTagger.Vectors.precalculate(csv)
    end
  end

  defp cached_download() do
    AudioTagger.SampleData.cache_dir()
    |> Path.join("icd9_codelist.csv")
  end
end
