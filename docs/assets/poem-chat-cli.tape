# Teff CLI demo: a poem workflow with two agents and human-in-the-loop
# approval (a classifier LLM understands free-form answers — no hard-coded
# keywords).
Output docs/assets/poem-chat-cli.gif
Set Width 1100
Set Height 660
Set FontSize 16
Set Padding 24
Set PlaybackSpeed 1.3

Type "source .venv/bin/activate"
Sleep 300ms
Enter
Sleep 700ms

Type `rm -rf examples/poem_chat/data`
Sleep 200ms
Enter
Sleep 400ms

Type `printf 'write a poem about autumn\nmake it shorter\nyes, perfect\n' | teff chat examples/poem_chat/workflow.yaml --session demo --prompt '> '`
Sleep 400ms
Enter
Sleep 45s