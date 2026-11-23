# The README demo as a GIF: `vhs docs/demo.tape` (https://github.com/charmbracelet/vhs).
# Needs a local Ollama with qwen2.5:7b and no METACULUS_TOKEN, so it uses the committed sample.
Output docs/demo.gif
Set FontSize 14
Set Width 1400
Set Height 800
Set TypingSpeed 40ms

Type "uv run augur models list"
Enter
Sleep 3s
Type "uv run augur forecast --tournament minibench"
Enter
Sleep 900s
Type "uv run augur ledger check augur.db"
Enter
Sleep 3s
