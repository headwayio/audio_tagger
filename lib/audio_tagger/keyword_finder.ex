defmodule AudioTagger.KeywordFinder do
  # def run(text) do
  #   phrases = extract_phrases(text)
  #   classify_text(text, phrases)
  #
  # # Example output:
  # # [
  # #   %{label: "known coronary artery disease", score: 0.35812485218048096},
  # #   %{label: "old man", score: 0.13899555802345276},
  # #   %{label: "comes", score: 0.12400370836257935},
  # #   %{label: "visit today", score: 0.10444625467061996},
  # #   %{label: "unstable angina", score: 0.09210294485092163}
  # # ]
  # end

  def prepare_zero_shot_classification_serving(labels) do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/bart-large-mnli"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-large-mnli"})

    Bumblebee.Text.zero_shot_classification(model_info, tokenizer, labels,
      compile: [batch_size: 1, sequence_length: 100],
      defn_options: [compiler: EXLA]
    )
  end

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

  def classify_text(text, labels) do
    serving = prepare_zero_shot_classification_serving(labels)
    output = Nx.Serving.run(serving, text)

    output.predictions
  end

  def extract_phrases(serving, text) do
    # Parts of speech to exclude from list of phrases

    output = Nx.Serving.run(serving, text)
    entities = output.entities

    cleanup_phrases(entities)
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
