defmodule AudioTagger.Utilities do
  @doc "Prints the elapsed time from the passed `time_start` prefixed with a label."
  def output_elapsed(label, time_start) do
    time_end = System.monotonic_time()

    IO.puts(
      "#{label}. Took #{System.convert_time_unit(time_end - time_start, :native, :millisecond)}ms"
    )
  end
end
