defmodule AudioTagger.Trainer do
  # AudioTagger.Trainer.train_model()

  @batch_size 32
  @sequence_length 64

  def start() do
    # Increase the backtrace depth for troubleshooting
    :erlang.system_flag(:backtrace_depth, 30)

    # 1. Load and configure the model
    {model_info, tokenizer} = prepare_model()
    %{model: model, params: params} = model_info

    # 2. Prepare data for training the model
    train_data = load_and_prepare_data(training_df(), tokenizer)
    test_data = load_and_prepare_data(test_df(), tokenizer)

    # 3. Gather small sets for CPU training
    train_data = Enum.take(train_data, 250)
    test_data = Enum.take(test_data, 50)

    # [{input, _}] = Enum.take(train_data, 1)
    # Axon.get_output_shape(model, input)

    # 4. Prepare and run training loop
    # logits_model = Axon.nx(model, & &1.logits)

    # Loss function from Bumblebee guide
    # loss =
    #   &Axon.Losses.categorical_cross_entropy(&1, &2,
    #     reduction: :mean,
    #     from_logits: true,
    #     sparse: true
    #   )

    # Assuming our data is of the format {sentence1, sentence2, percentage_float_of_similarity},
    # we may be able to use a cosine similarity loss function.
    loss = 
      &Axon.Losses.cosine_similarity(&1, &2) # , reduction: :mean)

    optimizer = Polaris.Optimizers.adamw(learning_rate: 2.0e-5)
    accuracy = &Axon.Metrics.accuracy(&1, &2, from_logits: true, sparse: true)

    # trained_model_state =
    #   logits_model
    #   |> Axon.Loop.trainer(loss, optimizer, log: 1)
    #   |> Axon.Loop.metric(accuracy, "accuracy")
    #   |> Axon.Loop.checkpoint(event: :epoch_completed)
    #   |> Axon.Loop.run(train_data, params, epochs: 3, compiler: EXLA, strict?: false)

    trained_model_state =
      model
      |> Axon.Loop.trainer(loss, optimizer, log: 1)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.checkpoint(event: :epoch_completed)
      |> Axon.Loop.run(train_data, params, epochs: 3, compiler: EXLA, strict?: false)

    # Serialize model to disk
    dir = Path.join(System.user_home(), "Library/Caches/audio_tagger")
    File.mkdir(dir)

    Axon.Loop.serialize_state(trained_model_state)
    |> File.write!(Path.join(dir, "trained_model_state.bin"))

    Axon.serialize(model, trained_model_state)
    |> File.write!(Path.join(dir, "serialized_model_with_trained_model_state.bin"))

    # Evaluate the model using our test data
    # logits_model
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(accuracy, "accuracy")
    |> Axon.Loop.run(test_data, trained_model_state, compiler: EXLA)
  end

  # Model usage:
  # search_input = Bumblebee.apply_tokenizer(tokenizer, [element])
  # search_embedding = Axon.predict(model_info.model, model_info.params, search_input, compiler: EXLA)
  defp prepare_model() do
    # labels = []
    model_definition = {:hf, "sentence-transformers/all-MiniLM-L6-v2"}

    # {:ok, spec} =
    #   Bumblebee.load_spec(model_definition,
    #     architecture: :for_sequence_classification
    #   )

    # - Previously we used a text classification model: You have a smallish set of labels that you want to inform the model of. (playing, dancing, cooking, travelling)
    # - Now we're using a sentence transformer / vector embedding model: In our case, we have thousands of labels (ICD-9 long descriptions) and also thousands of respective vector embeddings of those labels.
    # 
    # Questions:
    # - How do we configure the model to work with these labels?
    # - Do we use the original text labels or the computed vector embeddings of the labels?
    {:ok, spec} = Bumblebee.load_spec(model_definition)

    # num_labels = Enum.count(labels)

    # id_to_label =
    #   labels
    #   |> Enum.with_index(fn item, index -> {index, item} end)
    #   |> Enum.into(%{})

    # spec =
    #   Bumblebee.configure(spec, num_labels: num_labels, id_to_label: id_to_label)

    spec = Bumblebee.configure(spec)

    {:ok, model_info} = Bumblebee.load_model(model_definition, spec: spec)
    {:ok, tokenizer} = Bumblebee.load_tokenizer(model_definition)

    {model_info, tokenizer}
  end

  # defp load_and_prepare_data(df, tokenizer, opts \\ []) do
  defp load_and_prepare_data(df, tokenizer) do
    df
    |> stream()
    |> tokenize_and_batch(
      tokenizer,
      @batch_size,
      @sequence_length
      # opts[:id_to_label]
    )
  end

  defp stream(df) do
    anchor = df["text"] # ["abc", "def", ...]
    positive = df["long_description"] # ["qwerty", "asdfgh", ...]
    label = df["percentage_match"] # [1.0, 0.5, ...]

    # [{"abc", "qwerty", 1.0}, {"def", "asdfgh", 0.5}, ...]
    Stream.zip([
      Explorer.Series.to_enum(anchor),
      Explorer.Series.to_enum(positive), 
      Explorer.Series.to_enum(label)
    ])
  end

  # defp tokenize_and_batch(stream, tokenizer, batch_size, sequence_length, id_to_label) do
  defp tokenize_and_batch(stream, tokenizer, batch_size, sequence_length) do
    stream
    |> Stream.chunk_every(batch_size)
    |> Stream.map(fn batch ->
      # {anchor, positive, label} = Enum.unzip(batch)

      # {anchor, positive, label} = Enum.reduce(batch, {[], [], []}, fn {anchor_item, positive_item, label_item}, acc ->
      #   {anchor_list, positive_list, label_list} = acc
      #
      #   {
      #     [anchor_item] ++ anchor_list,
      #     [positive_item] ++ positive_list,
      #     [label_item] ++ label_list,
      #   }
      # end)

      {texts, label} = Enum.reduce(batch, {[], []}, fn {anchor_item, positive_item, label_item}, acc ->
        {texts_list, label_list} = acc

        {
          [{anchor_item, positive_item}] ++ texts_list,
          [label_item] ++ label_list,
        }
      end)

      # anchor = Enum.reverse(anchor)
      # positive = Enum.reverse(positive)
      texts = Enum.reverse(texts)
      label = Enum.reverse(label) |> Nx.tensor()

      # id_to_label_values = id_to_label |> Map.values()

      # label_ids =
      #   Enum.map(labels, fn item ->
      #     Enum.find_index(id_to_label_values, fn label_value -> label_value == item end)
      #   end)

      # tokenized_anchor = Bumblebee.apply_tokenizer(tokenizer, anchor, length: sequence_length)
      # tokenized_positive = Bumblebee.apply_tokenizer(tokenizer, positive, length: sequence_length)
      tokenized_texts = Bumblebee.apply_tokenizer(tokenizer, texts, length: sequence_length)
      # {tokenized, Nx.stack(label_ids)}

      # TODO: Is this the data format we want?
      # {tokenized_anchor, tokenized_positive, label}

      {tokenized_texts, Nx.stack(label)}
      # {Nx.Batch.concatenate([tokenized_texts]), Nx.stack(label)}
    end)
  end

  defp training_df() do
    "training_data_source.csv"
    |> Explorer.DataFrame.from_csv!()
    |> Explorer.DataFrame.select(["text", "long_description", "percentage_match"])
  end

  defp test_df() do
    "test_data_source.csv"
    |> Explorer.DataFrame.from_csv!()
    |> Explorer.DataFrame.select(["text", "long_description", "percentage_match"])
  end

  # def labels() do
  #   training_df()
  #   |> Explorer.DataFrame.distinct(["category"])
  #   |> Explorer.DataFrame.to_series()
  #   |> Map.get("category")
  #   |> Explorer.Series.to_list()
  # end
end
