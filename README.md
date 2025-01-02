<div align="center">
# 📝 SmartNote CLI 📝 
</div>

SmartNote is a CLI-first note-taking app with a touch of AI magic. 
I started this with the question if it would be easier to create full and complex AI apps if you started with CLI tools.  
Most of this is completely AI generated.
The result? A tool that organizes, summarizes, and supercharges your notes using GPT and embeddings.

## Features ✨

- **AI-Powered Note Management**
  - Generate titles and tags for your notes automatically 🏷️.
  - Analyze sentiment for emotional context 😊😡.
  - Highlight key points and suggest background knowledge 🔍.
- **Semantic Search & Relationships**
  - Notes are embedded with AI-generated vectors for finding related notes 🔗.
  - Visualize note relationships as a conceptual lattice 🕸️.
- **Voice Transcription** *(Prototype)* 🎙️
  - Transcribe audio notes and integrate them seamlessly into your database.
- **Summarization** 📋
  - Summarize all notes or generate context-specific summaries.
- **Friendly CLI Experience** 💻
  - Simple commands with colorful and formatted output.

---

## Installation 🚀

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ai-notes-cli.git
   cd ai-notes-cli
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python3 ai_notes_cli.py
   ```

4. **Set up your OpenAI API Key**:
   - When you first run the app, it will prompt you for your OpenAI API key 🔑.
   - Get your key from [OpenAI](https://platform.openai.com/signup).
   - The key will be securely stored in `~/.config/ai_notes/.apikey`.

---

## Usage 📖

### Initialize the Database 📦
Run this command to create the database:
```bash
python3 ai_notes_cli.py init_db
```

### Add a Note 📝
#### Text Note:
```bash
python3 ai_notes_cli.py add "This is my first note!"
```
#### Voice Note:
```bash
python3 ai_notes_cli.py add --voice path/to/audio/file.wav
```

### List Notes 📜
View all your notes:
```bash
python3 ai_notes_cli.py list
```
Use the `--fancy` flag for a visually pleasing output:
```bash
python3 ai_notes_cli.py list --fancy
```
Add context-specific summaries:
```bash
python3 ai_notes_cli.py list --context "meeting preparation"
```

### Summarize Notes ✍️
Generate a quick summary of all your notes:
```bash
python3 ai_notes_cli.py summarize
```

### View Concept Lattice 🕸️
Visualize conceptual relationships between your notes:
```bash
python3 ai_notes_cli.py lattice
```

---

## How It Works 🛠️

AI Notes CLI integrates the power of OpenAI models to:
- **Embed Notes**: Uses `text-embedding-ada-002` for vector embeddings to find semantic similarities.
- **ChatGPT**: Generates titles, tags, highlights, and sentiment analysis.
- **SQLite Database**: Stores all your notes in a lightweight local database for fast retrieval.

---

## Example Output 🎨

**Fancy Note Listing**:
```plaintext
╔══════════════════════════════════════════╗
║ Note #1 - My First AI Note               ║
╠══════════════════════════════════════════╝
Tags: ai, notes, experiment
Sentiment: positive
Key Highlight: This is an experiment to make note-taking smarter.
Background: AI-based tools can streamline workflows.
Related Notes: 2, 3
────────────────────────────────────────────
```

**Concept Lattice**:
```plaintext
ai: 1, 3
notes: 1, 2
experiment: 1
```

---

## Contributing 🤝

Got ideas or suggestions? Found a bug? Open an issue or submit a pull request! Contributions are always welcome. 🌟

---

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Support ❤️

If you find this project helpful, please ⭐ it on GitHub!

![Star Tracker](https://img.shields.io/github/stars/mmirman/SmartNote?style=social)

---

**Enjoy smarter, AI-enhanced note-taking! 🧠✨**
