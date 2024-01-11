defmodule AudioTagger.Classifier.TagResult do
  @enforce_keys [:label, :code, :score]
  defstruct [:label, :code, :score]
  @type t :: %__MODULE__{label: String.t(), code: String.t(), score: float()}
end
