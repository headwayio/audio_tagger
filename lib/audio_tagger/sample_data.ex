defmodule AudioTagger.SampleData do
  @cache_dir_suffix "Library/Caches/audio_tagger"
  @icd9_url "https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/downloads/icd-9-cm-v32-master-descriptions.zip"

  def cache_dir() do
    Path.join(System.user_home(), @cache_dir_suffix)
  end

  def get_icd9_code_list_csv() do
    # First, download and extract the ICD-9 text file source.
    download_icd9_code_list()

    # Then, create a CSV version of it.
    data = read_text_file()
    convert_text_to_csv(data)
  end

  defp download_icd9_code_list() do
    # TODO: These were brought in from an earlier Makefile. This could use Elixir functions instead of `cmd` for each
    # step.
    System.cmd("curl", [@icd9_url, "-o", "icd9_codelist.zip"], cd: cache_dir())

    System.cmd("unzip", ["-j", "icd9_codelist.zip", "CMS32_DESC_LONG_DX.txt"], cd: cache_dir())

    System.cmd("mv", ["CMS32_DESC_LONG_DX.txt", "icd9_codelist.txt"], cd: cache_dir())
  end

  defp read_text_file() do
    case cache_dir()
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
