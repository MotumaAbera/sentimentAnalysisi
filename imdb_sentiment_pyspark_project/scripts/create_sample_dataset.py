import csv
from pathlib import Path


positive_templates = [
    "A moving and beautifully acted film with excellent pacing and a memorable ending.",
    "Wonderful direction, strong performances, and a story that remains emotionally satisfying.",
    "The movie is charming, thoughtful, funny, and full of excellent character moments.",
    "A brilliant film with great music, impressive visuals, and confident storytelling.",
    "This was an amazing experience with warm humor and powerful acting throughout.",
    "The cast delivers outstanding performances and the screenplay feels fresh and heartfelt.",
    "A beautiful and inspiring movie that balances emotion, comedy, and drama very well.",
    "The plot is engaging, the direction is sharp, and the final scene is excellent.",
    "An enjoyable film with smart writing, strong chemistry, and wonderful atmosphere.",
    "A satisfying movie that earns its emotional moments and rewards patient viewers.",
]

negative_templates = [
    "The story was boring and the characters felt completely flat from beginning to end.",
    "I wanted to like it, but the plot was messy, dull, and poorly edited.",
    "The film wasted a good idea with weak writing, bad pacing, and forgettable acting.",
    "A disappointing movie with terrible dialogue and an ending that made little sense.",
    "The direction is unfocused, the jokes rarely work, and the drama feels forced.",
    "This was a slow and frustrating film with poor character development throughout.",
    "The movie feels empty, predictable, and far too long for such a thin story.",
    "A weak screenplay and awkward performances make the film difficult to enjoy.",
    "The visuals are fine, but the story is awful and the emotional scenes fail.",
    "A forgettable and boring experience that never becomes exciting or meaningful.",
]

rows = []
for i in range(12):
    for text in positive_templates:
        rows.append((f"{text} Review variation {i}.", "positive"))
    for text in negative_templates:
        rows.append((f"{text} Review variation {i}.", "negative"))

path = Path("data/imdb_reviews.csv")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["review", "sentiment"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} sample rows to {path}. Replace it with the full IMDb dataset for final results.")
