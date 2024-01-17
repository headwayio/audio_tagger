defmodule AudioTagger.Utilities do
  def output_elapsed(label, time_start) do
    time_end = System.monotonic_time()

    IO.puts(
      "#{label}. Took #{System.convert_time_unit(time_end - time_start, :native, :millisecond)}ms"
    )
  end
end
