defmodule AudioTagger.SampleData do
  @icd9_url "https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/downloads/icd-9-cm-v32-master-descriptions.zip"

  @doc "Returns a directory for storing temporary files. On macOS, this is within the user's Library folder."
  def cache_dir() do
    # Based on https://github.com/elixir-nx/bumblebee/blob/5c501b8e90f2cebf364708a06989fe41f148d99e/lib/bumblebee.ex#L1148
    if dir = System.get_env("AUDIO_TAGGER_CACHE_DIR") do
      Path.expand(dir)
    else
      :filename.basedir(:user_cache, "audio_tagger")
    end
  end

  @doc "Downloads the ICD-9 code list from the CMS web site and converts it to a CSV within `cache_dir()`."
  def get_icd9_code_list_csv(current_directory \\ cache_dir()) do
    unless File.exists?(current_directory) do
      File.mkdir!(current_directory)
    end

    # First, download and extract the ICD-9 text file source.
    download_icd9_code_list(current_directory)

    # Then, create a CSV version of it.
    data = read_text_file(current_directory)
    convert_text_to_csv(data)
  end

  defp download_icd9_code_list(current_directory) do
    zip = "icd9_codelist.zip"
    source = "CMS32_DESC_LONG_DX.txt"
    destination = "icd9_codelist.txt"

    # TODO: These were brought in from an earlier Makefile. This could use Elixir functions instead of `cmd` for each
    # step.
    System.cmd("curl", [@icd9_url, "-o", zip], cd: current_directory)
    System.cmd("unzip", ["-j", zip, source], cd: current_directory)
    System.cmd("mv", [source, destination], cd: current_directory)
    # File.rename!(
    #   Path.join(current_directory, source),
    #   Path.join(current_directory, destination)
    # )
  end

  defp read_text_file(current_directory) do
    case current_directory
         |> Path.join("icd9_codelist.txt")
         |> File.read() do
      {:ok, data} -> data
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
      (["\"code\",\"long_description\""] ++ csv_data)
      |> Enum.join("\n")
      |> String.replace_invalid()

    case cache_dir()
         |> Path.join("icd9_codelist.csv")
         |> File.write(csv_data) do
      :ok -> :ok
      {:error, error} -> IO.puts("Failed to write converted CSV file: #{error}")
    end
  end
end
