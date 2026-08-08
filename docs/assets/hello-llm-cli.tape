# Teff CLI demo: run an LLM workflow with a checkpointer, then graph it.
Output docs/assets/hello-llm-cli.gif
Set Width 1000
Set Height 520
Set FontSize 16
Set Padding 24
Set PlaybackSpeed 1.4

Type "source .venv/bin/activate"
Sleep 500ms
Enter
Sleep 1.5s

Type `teff run --file examples/hello_llm/workflow.yaml --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id hello`
Sleep 500ms
Enter
Sleep 9s

Type `teff run --file examples/hello_llm/workflow.yaml --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id hello --trace`
Sleep 500ms
Enter
Sleep 6s

Type `teff graph examples/hello_llm/workflow.yaml`
Sleep 500ms
Enter
Sleep 5s