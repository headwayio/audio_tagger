defmodule AudioTagger.Structs.TagResult do
  @moduledoc """
  Represents a single result of the classification process. Each initial input text may have multiple TagResults.

  - label: The `long_description` field
  - code: The `code` field
  - score: How similar the label is to the input text on a scale from 0.0 to 1.0
  """
  @enforce_keys [:label, :code, :score]
  defstruct [:label, :code, :score]
  @type t :: %__MODULE__{label: String.t(), code: String.t(), score: float()}
end
