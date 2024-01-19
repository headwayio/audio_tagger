defmodule AudioTagger.Classifier do
  @moduledoc """
  Provides shared utilities for classification strategies.
  """

  require Explorer.DataFrame

  @doc "Looks up a `description` within a dataframe to find its associated code"
  def code_for_label(labels_df, description) do
    Explorer.DataFrame.filter(labels_df, long_description == ^description)
    |> Explorer.DataFrame.pull("code")
    |> Explorer.Series.first()
  end

  @doc "Retrieves a list of `long_description` values from a dataframe"
  def to_list_of_label_descriptions(labels_df) do
    labels_df
    |> Explorer.DataFrame.pull("long_description")
    |> Explorer.Series.to_list()
  end
end
