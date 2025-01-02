#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import sqlite3
import json
from pathlib import Path
from colorama import Fore, Style, init
import openai
import math

init(autoreset=True)

# Constants for API key handling
CONFIG_DIR = Path.home() / ".config" / "ai_notes"
API_KEY_FILE = CONFIG_DIR / ".apikey"

###########################
# OpenAI API Key Handling
###########################

def get_api_key():
    """Fetch the OpenAI API key from the configuration file."""
    if API_KEY_FILE.exists():
        with open(API_KEY_FILE, "r") as file:
            return file.read().strip()
    else:
        return None

def set_api_key():
    """Prompt the user to enter their OpenAI API key and save it to the configuration file."""
    print(Fore.YELLOW + "It seems you don't have an OpenAI API key configured.")
    print("You can get an API key from: " + Fore.CYAN + "https://platform.openai.com/signup")
    api_key = input(Fore.GREEN + "Please enter your OpenAI API key: ").strip()
    
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(API_KEY_FILE, "w") as file:
        file.write(api_key)
    print(Fore.GREEN + f"API key saved to {API_KEY_FILE}")
    return api_key

def initialize_openai_api():
    openai.api_key = get_api_key()
    if not openai.api_key:
        openai.api_key = set_api_key()

initialize_openai_api()

DB_FILE = Path.home() / ".ai_notes_real.db"

#########################
# Database Setup & Utils
#########################

def get_db():
    return sqlite3.connect(DB_FILE)

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        title TEXT,
        tags TEXT,
        sentiment TEXT,
        highlight TEXT,
        background TEXT,
        timestamp TEXT,
        embedding BLOB
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS related_notes (
        note_id INTEGER,
        related_note_id INTEGER
    )""")
    conn.commit()
    conn.close()

def store_note(note_data):
    # note_data keys: text, title, tags, sentiment, highlight, background, timestamp, embedding (bytes)
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO notes (text, title, tags, sentiment, highlight, background, timestamp, embedding) VALUES (?,?,?,?,?,?,?,?)",
              (note_data["text"], note_data["title"], ",".join(note_data["tags"]), note_data["sentiment"], 
               note_data["highlight"], note_data["background"], note_data["timestamp"], note_data["embedding"]))
    note_id = c.lastrowid
    # Store related notes
    for rnid in note_data.get("related_ids", []):
        c.execute("INSERT INTO related_notes (note_id, related_note_id) VALUES (?,?)", (note_id, rnid))
    conn.commit()
    conn.close()
    return note_id

def get_all_notes():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, text, title, tags, sentiment, highlight, background, timestamp, embedding FROM notes ORDER BY id")
    rows = c.fetchall()
    conn.close()
    notes = []
    for r in rows:
        notes.append({
            "id": r[0],
            "text": r[1],
            "title": r[2],
            "tags": r[3].split(",") if r[3] else [],
            "sentiment": r[4],
            "highlight": r[5],
            "background": r[6],
            "timestamp": r[7],
            "embedding": r[8]
        })
    return notes

def get_related_notes_map():
    # returns a dict { note_id: [related_ids] }
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT note_id, related_note_id FROM related_notes")
    rows = c.fetchall()
    conn.close()
    rel_map = {}
    for nid, rid in rows:
        rel_map.setdefault(nid, []).append(rid)
    return rel_map

#############################
# OpenAI / AI Util Functions
#############################

def gpt_chat(prompt, system="", model="gpt-4", temperature=0.7):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(Fore.RED + f"Error calling OpenAI: {e}")
        return ""

def get_embeddings(text):
    # Use text-embedding-ada-002
    try:
        res = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return res["data"][0]["embedding"]
    except Exception as e:
        print(Fore.RED + f"Embedding API error: {e}")
        return []

def vector_to_blob(vec):
    # Store embedding as JSON string
    return json.dumps(vec).encode('utf-8')

def blob_to_vector(blob):
    return json.loads(blob.decode('utf-8'))

def analyze_note_for_title_and_tags(note_text):
    prompt = f"Suggest a concise title and a few tags for this note:\n\n{note_text}\nFormat:\nTitle: ...\nTags: tag1, tag2"
    res = gpt_chat(prompt)
    title = "Untitled"
    tags = []
    for line in res.split('\n'):
        line = line.strip()
        if line.lower().startswith("title:"):
            title = line.split(":",1)[1].strip()
        elif line.lower().startswith("tags:"):
            tags_line = line.split(":",1)[1].strip()
            tags = [t.strip().lower() for t in tags_line.split(",")]
    return title, tags

def mood_modeling(note_text):
    prompt = f"Analyze the emotional tone of the following note and respond with a single descriptive word:\n{note_text}"
    return gpt_chat(prompt, temperature=0)

def highlight_key_points(note_text):
    prompt = f"From this note:\n{note_text}\nHighlight the most important sentence or phrase in one short sentence."
    return gpt_chat(prompt)

def ambient_knowledge_integration(note_text):
    prompt = f"Given this note:\n{note_text}\nProvide a brief background fact or link to enhance understanding."
    return gpt_chat(prompt, temperature=0.7)

def find_related_notes(new_embedding, all_notes):
    # Compute cosine similarity and return top related notes
    # We'll choose top 2-3 related notes
    new_vec = new_embedding
    scores = []
    for n in all_notes:
        if not n["embedding"]:
            continue
        vec = blob_to_vector(n["embedding"])
        score = cosine_similarity(new_vec, vec)
        if score > 0.8: # arbitrary threshold for "related"
            scores.append((n["id"], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores[:3]]

def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a*norm_b == 0:
        return 0
    return dot/(norm_a*norm_b)

def quick_summary_of_all(notes):
    if not notes:
        return "No notes available."
    all_text = "\n".join([n["text"] for n in notes])
    prompt = f"Summarize these notes in a few bullet points:\n{all_text}"
    return gpt_chat(prompt)

def situation_aware_summary(notes, context):
    if not notes:
        return "No notes yet."
    all_text = "\n".join([n["text"] for n in notes])
    prompt = f"Summarize these notes in a way that is particularly useful for a {context} scenario.\nNotes:\n{all_text}"
    return gpt_chat(prompt)

def simulate_voice_transcription(file_path):
    # Placeholder for transcription. In reality, integrate Whisper or another STT.
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return "Transcribed from voice: " + f.read()
    return "Transcribed from voice (placeholder)."

def build_concept_lattice(notes):
    # We'll simulate a concept lattice by grouping tags
    # A full concept lattice is complex; we’ll just group by tags.
    tag_map = {}
    for n in notes:
        for t in n["tags"]:
            tag_map.setdefault(t, []).append(n["id"])
    return tag_map

def display_concept_lattice(notes):
    tag_map = build_concept_lattice(notes)
    if not tag_map:
        print("No concepts extracted.")
        return
    print(Fore.CYAN + "Concept Lattice (Approximated by Tags):")
    for concept, ids in tag_map.items():
        line = f"{concept}: " + ", ".join(str(i) for i in ids)
        print(line)

def display_notes(notes, fancy=False, context=None):
    if context:
        print(Fore.CYAN + f"Contextual Summary for {context}:")
        print(situation_aware_summary(notes, context))
        print()

    if not notes:
        print("No notes yet.")
        return
    
    rel_map = get_related_notes_map()
    for note in notes:
        related_ids = rel_map.get(note["id"], [])
        display_note(note, related_ids, fancy)

    if fancy:
        visualize_related_notes(rel_map)

def display_note(note, related_ids, fancy=False):
    title = note["title"]
    tags = ", ".join(note["tags"])
    sentiment = note["sentiment"] or "neutral"
    highlight = note.get("highlight", "")
    if fancy:
        print(Fore.MAGENTA + "╔" + "═"*50 + "╗")
        header = f"Note #{note['id']} - {title}"
        print(Fore.MAGENTA + "║ " + Fore.CYAN + header + Fore.MAGENTA + " "*(50 - len(header)) + "║")
        print(Fore.MAGENTA + "╠" + "═"*50 + "╝")
        print(Fore.WHITE + f"Tags: {tags}")
        print(Fore.YELLOW + f"Sentiment: {sentiment}")
        if highlight:
            print(Fore.GREEN + f"Key Highlight: {highlight}")
        if related_ids:
            print(Fore.BLUE + f"Related Notes: {', '.join(str(x) for x in related_ids)}")
        if note["background"]:
            print(Fore.CYAN + f"Background: {note['background']}")
        print(Fore.MAGENTA + "─"*52)
    else:
        print(f"Note #{note['id']}: {title}")
        print(f"  Tags: {tags}")
        print(f"  Sentiment: {sentiment}")
        if highlight:
            print(f"  Key Highlight: {highlight}")
        if related_ids:
            print(f"  Related Notes: {', '.join(str(x) for x in related_ids)}")
        if note["background"]:
            print(f"  Background: {note['background']}")
        print("-"*60)

def visualize_related_notes(rel_map):
    if not rel_map:
        print("No inter-note relationships found.")
        return
    print(Fore.CYAN + "Notes Relationship Graph:")
    for nid, rids in rel_map.items():
        for rid in rids:
            print(f"{nid} --> {rid}")
    print()


def remove_note_by_id(note_id):
    conn = get_db()
    c = conn.cursor()
    # Remove from notes
    c.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    # Remove from related_notes
    c.execute("DELETE FROM related_notes WHERE note_id = ? OR related_note_id = ?", (note_id, note_id))
    conn.commit()
    conn.close()

def update_note(note_id, updated_data):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        UPDATE notes
        SET text = ?, title = ?, tags = ?, sentiment = ?, highlight = ?, background = ?, timestamp = ?, embedding = ?
        WHERE id = ?
    """, (
        updated_data["text"], updated_data["title"], ",".join(updated_data["tags"]), updated_data["sentiment"],
        updated_data["highlight"], updated_data["background"], updated_data["timestamp"], updated_data["embedding"],
        note_id
    ))
    conn.commit()
    conn.close()

def get_note_by_id(note_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, text, title, tags, sentiment, highlight, background, timestamp, embedding FROM notes WHERE id = ?", (note_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "text": row[1],
        "title": row[2],
        "tags": row[3].split(",") if row[3] else [],
        "sentiment": row[4],
        "highlight": row[5],
        "background": row[6],
        "timestamp": row[7],
        "embedding": row[8]
    }

#########################
# Semantic Search Feature
#########################

def semantic_search(query, all_notes):
    query_embedding = get_embeddings(query)
    results = []
    for note in all_notes:
        if not note["embedding"]:
            continue
        note_vec = blob_to_vector(note["embedding"])
        score = cosine_similarity(query_embedding, note_vec)
        results.append((note, score))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return [note for note, score in results if score > 0.7]  # Arbitrary threshold

#########################
# Newsfeed Feature
#########################

def notes_by_date(target_date):
    target_date = target_date.isoformat()[:10]
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, text, title, tags, sentiment, highlight, background, timestamp, embedding FROM notes WHERE timestamp LIKE ?", (f"{target_date}%",))
    rows = c.fetchall()
    conn.close()
    notes = []
    for r in rows:
        notes.append({
            "id": r[0],
            "text": r[1],
            "title": r[2],
            "tags": r[3].split(",") if r[3] else [],
            "sentiment": r[4],
            "highlight": r[5],
            "background": r[6],
            "timestamp": r[7],
            "embedding": r[8]
        })
    return notes

#####################
# CLI Enhancements
#####################

def main():
    parser = argparse.ArgumentParser(description="AI-First Note-Taking App")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Initialization
    init_parser = subparsers.add_parser("init_db", help="Initialize the database")

    # Add Note
    add_parser = subparsers.add_parser("add", help="Add a new note")
    add_parser.add_argument("text", nargs="*", help="The note text")
    add_parser.add_argument("--voice", help="Path to an audio file to transcribe")

    # View Notes
    list_parser = subparsers.add_parser("list", help="List all notes")
    list_parser.add_argument("--fancy", action="store_true", help="Fancy display")
    list_parser.add_argument("--context", help="Context for a situational summary")

    # New Commands
    subparsers.add_parser("summarize", help="Summarize all notes")
    subparsers.add_parser("lattice", help="Show conceptual lattice")

    # Feature: Read a note
    read_parser = subparsers.add_parser("read", help="Read a specific note by ID")
    read_parser.add_argument("id", type=int, help="The ID of the note to read")

    # Feature: Semantic search
    search_parser = subparsers.add_parser("search", help="Search notes by semantic similarity")
    search_parser.add_argument("query", help="The query for semantic search")

    # Feature: Remove note
    remove_parser = subparsers.add_parser("remove", help="Remove a note by ID")
    remove_parser.add_argument("id", type=int, help="The ID of the note to remove")

    # Feature: Edit note
    edit_parser = subparsers.add_parser("edit", help="Edit a note by ID")
    edit_parser.add_argument("id", type=int, help="The ID of the note to edit")
    edit_parser.add_argument("text", nargs="*", help="The new text for the note")

    # Feature: Newsfeed
    newsfeed_parser = subparsers.add_parser("newsfeed", help="View notes for a specific date")
    newsfeed_parser.add_argument("date", help="The date in YYYY-MM-DD format (e.g., 2025-01-02)")

    args = parser.parse_args()

    if args.command == "init_db":
        init_db()
        print("Database initialized.")
        return

    if args.command == "read":
        note = get_note_by_id(args.id)
        if note:
            related_map = get_related_notes_map()
            related_ids = related_map.get(note["id"], [])
            display_note(note, related_ids, fancy=True)
        else:
            print(Fore.RED + f"No note found with ID {args.id}.")
        return

    if args.command == "search":
        notes = get_all_notes()
        results = semantic_search(args.query, notes)
        if results:
            print(Fore.GREEN + f"Results for '{args.query}':")
            display_notes(results, fancy=True)
        else:
            print(Fore.YELLOW + f"No relevant notes found for query: {args.query}")
        return

    if args.command == "remove":
        remove_note_by_id(args.id)
        print(Fore.GREEN + f"Note #{args.id} removed.")
        return

    if args.command == "edit":
        note = get_note_by_id(args.id)
        if not note:
            print(Fore.RED + f"No note found with ID {args.id}.")
            return

        new_text = " ".join(args.text).strip()
        if not new_text:
            print(Fore.YELLOW + "No new text provided. Exiting.")
            return

        # Recompute metadata
        title, tags = analyze_note_for_title_and_tags(new_text)
        sentiment = mood_modeling(new_text)
        highlight = highlight_key_points(new_text)
        background = ambient_knowledge_integration(new_text)
        embedding = vector_to_blob(get_embeddings(new_text))

        updated_data = {
            "text": new_text,
            "title": title,
            "tags": tags,
            "sentiment": sentiment,
            "highlight": highlight,
            "background": background,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "embedding": embedding
        }

        update_note(args.id, updated_data)
        print(Fore.GREEN + f"Note #{args.id} updated.")
        return

    if args.command == "newsfeed":
        try:
            target_date = datetime.datetime.strptime(args.date, "%Y-%m-%d")
            notes = notes_by_date(target_date)
            if notes:
                print(Fore.CYAN + f"Notes for {args.date}:")
                display_notes(notes, fancy=True)
            else:
                print(Fore.YELLOW + f"No notes found for {args.date}.")
        except ValueError:
            print(Fore.RED + "Invalid date format. Use YYYY-MM-DD.")
        return

    if args.command == "init_db":
        init_db()
        print("Database initialized.")
        return

    if not DB_FILE.exists():
        print(Fore.RED + "Database not found. Run 'init_db' first.")
        return

    if args.command == "add":
        if args.voice:
            note_text = simulate_voice_transcription(args.voice)
        else:
            note_text = " ".join(args.text).strip()
            if not note_text:
                note_text = input("Enter your note text: ")

        # Analyze note
        title, tags = analyze_note_for_title_and_tags(note_text)
        sentiment = mood_modeling(note_text)
        highlight = highlight_key_points(note_text)
        background = ambient_knowledge_integration(note_text)
        embedding = get_embeddings(note_text)

        all_notes = get_all_notes()
        related_ids = find_related_notes(embedding, all_notes)

        note_data = {
            "text": note_text,
            "title": title,
            "tags": tags,
            "sentiment": sentiment,
            "highlight": highlight,
            "background": background,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "embedding": vector_to_blob(embedding),
            "related_ids": related_ids
        }

        note_id = store_note(note_data)
        print(f"Added note #{note_id} titled '{title}' with sentiment '{sentiment}'.")

    elif args.command == "list":
        notes = get_all_notes()
        display_notes(notes, fancy=args.fancy, context=args.context)

    elif args.command == "summarize":
        notes = get_all_notes()
        summary = quick_summary_of_all(notes)
        print("Summary of All Notes:")
        print(summary)

    elif args.command == "lattice":
        notes = get_all_notes()
        display_concept_lattice(notes)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
