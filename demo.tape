Output demo.gif

Require Echo

Set FontSize 14
Set Width 900
Set Height 500
Set Theme "Dracula"
Set Shell "bash"
Set PlaybackSpeed 1.25

Env INFER_URL "http://192.168.4.30:11434/v1"
Env INFER_MODEL "gemma4:latest"
Env INFER_API_KEY "ollama"

Sleep 1s
Type `infer "what directory am i in"`
Sleep 500ms
Enter
Sleep 8s

Sleep 1s
Type `echo "TypeError: Cannot read property 'id' of undefined" | infer "one-line fix"`
Sleep 500ms
Enter
Sleep 12s

Sleep 1s
Type `infer -j '{"name":"string","pid":0}' "current process info"`
Sleep 500ms
Enter
Sleep 8s

Sleep 2s
