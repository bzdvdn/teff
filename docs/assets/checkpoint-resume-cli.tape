# Teff demo: crash mid-LLM run, resume from the failed node.
Output docs/assets/checkpoint-resume-cli.gif
Set Width 1000
Set Height 520
Set FontSize 16
Set Padding 24
Set PlaybackSpeed 1.3

Type "source .venv/bin/activate"
Sleep 500ms
Enter
Sleep 1.5s

Type "TEEF_KEEP_CHECKPOINT=1 python examples/checkpoint_resume/main.py"
Sleep 500ms
Enter
Sleep 12s

Type `teff inspect --checkpoint '{"type":"sqlite","path":"examples/checkpoint_resume/checkpoints.db"}' --checkpoint-id durable-run`
Sleep 500ms
Enter
Sleep 3s
