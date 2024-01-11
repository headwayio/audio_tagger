defmodule Mix.Tasks.Embeddings do
  @moduledoc "Generate embeddings from ICD-9 codes"

  use Mix.Task

  @shortdoc "Generates icd10_vector_tensors.bin in system tmp"
  def run(args) do
    Mix.Task.run("app.start")
    csv = Enum.at(args, 0)

    AudioTagger.Vectors.precalculate(csv)
  end
end
