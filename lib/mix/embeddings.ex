defmodule Mix.Tasks.Embeddings do
  @moduledoc "Generate embeddings from downloaded ICD-9 code list"

  use Mix.Task

  @shortdoc "Generates icd10_vector_tensors.bin in system tmp"
  def run(args) do
    Mix.Task.run("app.start")

    csv = Enum.at(args, 0) || "icd9_codelist.csv"

    unless File.exists?(csv) do
      retrieve_codelist()
    end

    AudioTagger.Vectors.precalculate(csv)
  end

  defp retrieve_codelist() do
    # Call out to makefile to download ICD-9 data and unzip the necessary file.
    System.cmd("make", ["all"])

    # If the file is found, create a CSV version of it.
    case File.read("icd9_codelist.txt") do
      {:ok, data} -> convert_text_to_csv(data)
      {:error, error} -> IO.puts("Failed to read icd9_codelist.txt file: #{error}")
    end
  end

  defp convert_text_to_csv(data) do
    csv_data =
      data
      |> String.split("\n")
      |> Enum.map(fn line ->
        split = String.split(line, " ", parts: 2)
        code = Enum.at(split, 0)
        long_description = Enum.at(split, 1)

        description =
          case is_binary(long_description) do
            true -> String.trim(long_description)
            false -> ""
          end

        "\"#{code}\",\"#{description}\""
      end)

    csv_data =
      (["\"CODE\",\"LONG DESCRIPTION\""] ++ csv_data)
      |> Enum.join("\n")

    case File.write("icd9_codelist.csv", csv_data) do
      :ok -> :ok
      {:error, error} -> IO.puts("Failed to write converted CSV file: #{error}")
    end
  end
end
