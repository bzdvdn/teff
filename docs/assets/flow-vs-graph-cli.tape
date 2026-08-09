# Teff API demo: the same workflow at two levels — the Flow builder and the
# low-level Graph API — then the compiled graph rendered as Mermaid.
Output docs/assets/flow-vs-graph-cli.gif
Set Width 1000
Set Height 520
Set FontSize 16
Set Padding 24
Set PlaybackSpeed 1.3

Type "source .venv/bin/activate"
Sleep 500ms
Enter
Sleep 1.5s

Type "python examples/hello_workflow/flow_dsl.py"
Sleep 500ms
Enter
Sleep 4s

Type "python examples/hello_workflow/low_level.py"
Sleep 500ms
Enter
Sleep 4s

Type "teff graph examples/hello_workflow/workflow.yaml --mermaid"
Sleep 500ms
Enter
Sleep 5s
