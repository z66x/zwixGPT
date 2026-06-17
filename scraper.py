import os
import re
import json
import lyricsgenius

GENIUS_TOKEN = os.environ.get("GENIUS_TOKEN")

artists = [
    "Kanye West", "Kendrick Lamar", "Drake", "Eminem",
    "Travis Scott", "J. Cole", "XXXTentacion", "NF"
]

output_folder = "rap_dataset"
os.makedirs(output_folder, exist_ok=True)

def clean_lyrics(lyrics):
    lyrics = re.sub(r'\[.*?\]', '', lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
    lyrics = re.sub(r'\d*[KM]?Embed$', '', lyrics)
    return lyrics.strip()

genius = lyricsgenius.Genius(
    GENIUS_TOKEN,
    skip_non_songs=True,      
    excluded_terms=["(Remix)", "(Live)", "(Demo)"],  
    remove_section_headers=True,
    timeout=15,
    retries=3
)
genius.verbose = True

seen_songs = set()

# Pre-load the tracker with any songs downloaded on previous days
print("Scanning existing files for duplicates...")
for filename in os.listdir(output_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(output_folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for song in existing_data:
                    seen_songs.add(song["title"].lower().strip())
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            
print(f"Memory loaded. Currently avoiding {len(seen_songs)} known duplicates.\n")


for artist_name in artists:
    file_path = os.path.join(output_folder, f"{artist_name}.json")
    if os.path.exists(file_path):
        print(f"[{artist_name}] File already exists. Skipping.")
        continue

    print(f"[{artist_name}] Sending request to server for top 100 songs...")
    try:
        artist = genius.search_artist(artist_name, max_songs=100, sort="popularity")
        
        if artist:
            total_songs = len(artist.songs)
            print(f"[{artist_name}] Received response. Checking {total_songs} songs...")
            
            artist_dataset = []
            saved_count = 0  # Counter only goes up on successful save
            
            for song in artist.songs:
                clean_title = song.title.lower().strip()
                
                if clean_title not in seen_songs and song.lyrics:
                    cleaned = clean_lyrics(song.lyrics)
                    if len(cleaned) > 100:  
                        saved_count += 1
                        artist_dataset.append({
                            "artist": artist_name,
                            "title": song.title,
                            "lyrics": cleaned
                        })
                        seen_songs.add(clean_title)
                        
                        # Explicitly state it was ADDED
                        print(f"[{artist_name}] Added {saved_count} | Title: '{song.title}'")
                    else:
                        print(f"[{artist_name}] -> Skipped '{song.title}': Lyrics too short after cleaning.")
                else:
                    # Explicitly state WHICH song was skipped
                    print(f"[{artist_name}] -> Skipped '{song.title}': Duplicate or missing lyrics.")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(artist_dataset, f, indent=4)
            
            print(f"[{artist_name}] Successfully saved {saved_count} clean songs.\n")

    except Exception as e:
        print(f"[{artist_name}] ERROR: Something went wrong - {e}\n")
        
print("Script finished.")