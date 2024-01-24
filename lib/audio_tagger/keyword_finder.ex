defmodule AudioTagger.KeywordFinder do
  @moduledoc """
  Provides Nx.Servings and a utility function for finding relevant phrases in text and then classifying the text with 
  those phrases. For example:
  1. Input text:
     "Last month he was admitted to our hospital with unstable angina."
  2. Run token classification + keep only verbs and adjective + noun combinations:
     ["last month", "admitted", "hospital", "untable angina"]
  3. Then, take the original input text in #1 and run zero-shot classification with the labels in #2:
     [
       %{label: "unstable angina", score: 0.433},
       %{label: "last month", score: 0.276},
       %{label: "admitted", score: 0.212},
       %{label: "hospital", score: 0.079},
     ]
  """

  @doc """
  Creates an Nx.Serving for classifying tokens within text. The result of running this serving is a map including
  `entities` in this format:
  ```
    [
      %{label: "ADJ", start: 0, end: 4, score: 0.9987151622772217, phrase: "last"},
      %{label: "NOUN", start: 5, end: 10, score: 0.999144434928894, phrase: "month"},
      %{label: "PRON", start: 11, end: 13, score: 0.9995080232620239, phrase: "he"},
      %{label: "AUX", start: 14, end: 17, score: 0.9972853660583496, phrase: "was"},
      %{label: "VERB", start: 18, end: 26, score: 0.9994298815727234, phrase: "admitted"},
      ...
    ]
  ```

  Can be paired with `cleanup_phrases` to combine adjectives with following nouns (e.g. "last month" as a single value
  instead of "last" and "month") and keep only those combinations and the verbs.
  """
  def prepare_token_classification_serving() do
    {:ok, model_info} =
      Bumblebee.load_model({:hf, "vblagoje/bert-english-uncased-finetuned-pos"})

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

    Bumblebee.Text.token_classification(model_info, tokenizer,
      aggregation: :same,
      compile: [batch_size: 1, sequence_length: 100],
      defn_options: [compiler: EXLA]
    )
  end

  @doc """
  Creates an Nx.Serving for classifying text based on the passed `labels`. The result of running this serving is a map
  including `predications` in this format:
  ```
    [
      %{label: "unstable angina", score: 0.433},
      %{label: "last month", score: 0.276},
      %{label: "admitted", score: 0.212},
      %{label: "hospital", score: 0.079},
    ]
  ```
  """
  def prepare_zero_shot_classification_serving(labels) do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})

    Bumblebee.Text.zero_shot_classification(model_info, tokenizer, labels,
      compile: [batch_size: 1, sequence_length: 100],
      defn_options: [compiler: EXLA]
    )
  end

  def find_most_similar_label(text, labels, max_k \\ 5) do
    label_embeddings = AudioTagger.Vectors.embed(labels)
    search_embedding = AudioTagger.Vectors.embed([text])

    k = min(Enum.count(labels), max_k)

    {values, indices_of_most_similar} =
      Bumblebee.Utils.Nx.cosine_similarity(
        search_embedding.pooled_state,
        label_embeddings.pooled_state
      )
      |> Nx.top_k(k: k)

    List.zip([
      Nx.to_flat_list(indices_of_most_similar),
      Nx.to_flat_list(values)
    ])
    |> Enum.map(fn {index, score} ->
      %{
        label: Enum.at(labels, index),
        score: score
      }
    end)
  end

  def cleanup_phrases(entities) do
    ignored = ["DET", "PUNCT", "ADP", "NUM", "AUX", "PRON"]

    entities
    |> Enum.reduce([], fn entity, acc ->
      if Enum.member?(ignored, entity.label) do
        acc
      else
        # This leaves "VERB", "NOUN", and "ADJ"
        # If the label is an adjective, combine it with the next label.
        next_phrase =
          if entity.label == "ADJ" do
            "#{entity.phrase} [CONTINUATION]"
          else
            entity.phrase
          end

        if Enum.count(acc) > 0 do
          previous = Enum.at(acc, -1)

          # First, check if the previous phrase ends with a continuation token.  
          if String.ends_with?(previous, "[CONTINUATION]") do
            acc_without_last = Enum.take(acc, Enum.count(acc) - 1)
            acc_without_last ++ [String.replace(previous, "[CONTINUATION]", next_phrase)]
          else
            acc ++ [next_phrase]
          end
        else
          acc ++ [next_phrase]
        end
      end
    end)
    |> Enum.map(fn phrase -> String.replace(phrase, " [CONTINUATION]", "") end)
  end
end
