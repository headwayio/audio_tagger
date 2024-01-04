defmodule AudioTagger.MixProject do
  use Mix.Project

  @version "0.1.0"
  @description "Provides utilities to transcribe an audio recording and tag portions based on a list of provided terms."

  def project do
    [
      app: :audio_tagger,
      version: @version,
      description: @description,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:kino, "~> 0.11.3"},
      {:kino_bumblebee, "~> 0.4.0"},
      {:exla, ">= 0.0.0"},
      {:explorer, "~> 0.7.0"},
      {:kino_explorer, "~> 0.1.11"}

      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end