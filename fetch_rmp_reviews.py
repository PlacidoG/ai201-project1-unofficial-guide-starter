

import re
import json
from pathlib import Path
import RateMyProfessor_Database_APIs as rmp

prof_links = [
    "https://www.ratemyprofessors.com/professor/2633588",
    "https://www.ratemyprofessors.com/professor/2690400",
    "https://www.ratemyprofessors.com/professor/2889570",
    "https://www.ratemyprofessors.com/professor/2051130",
    "https://www.ratemyprofessors.com/professor/2179343",
    "https://www.ratemyprofessors.com/professor/173959",
    "https://www.ratemyprofessors.com/professor/2361808",
    "https://www.ratemyprofessors.com/professor/2918727",
    "https://www.ratemyprofessors.com/professor/549688",
    "https://www.reddit.com/r/uhd/comments/18de5n2/cs_classes_prof_info_please/"
]


id_pattern = re.compile(r"/professor/(\d+)")

def extract_prof_id(url: str) -> int:
    m = id_pattern.search(url)
    if not m:
        raise ValueError(f"Could not extract professor ID from URL: {url}")
    return int(m.group(1))

def fetch_professor(prof_id: int):
    try:
        prof = rmp.fetch_a_professor(prof_id)
        return prof
    except Exception as e:
        print(f"Error fetching professor {prof_id}: {e}")
        return None

for url in prof_links:
    prof_id = extract_prof_id(url)
    prof = fetch_professor(prof_id)
    print(f"\n--- Professor {prof_id} from {url} ---")
    if prof is None:
        print("Failed to fetch")
        continue
    # Inspect structure once so you know where reviews live
    if isinstance(prof, dict):
        print(prof.keys())  # see top‑level keys
        # optionally print a small sample
        print(json.dumps(prof, indent=2)[:2000])
    else:
        print(prof)
    break  # remove this once you understand the structure

def extract_reviews_from_prof(prof: dict, source_url: str):
    """
    Return a list of normalized review dicts for one professor.
    Adjust keys based on actual structure from your print() inspection.
    """
    reviews = []

    # Example guesses – replace "reviews" and inner keys with real ones you see.
    raw_reviews = prof.get("reviews") or prof.get("ratings") or []

    for r in raw_reviews:
        reviews.append({
            "professor_name": prof.get("professor_name") or prof.get("name"),
            "professor_id": prof.get("professor_id"),
            "school_name": prof.get("school_name"),
            "school_id": prof.get("school_id"),
            "source_url": source_url,
            "course": r.get("course") or r.get("course_name"),
            "rating_overall": r.get("rating"),
            "difficulty": r.get("difficulty"),
            "date": r.get("date"),
            "review_text": r.get("comment") or r.get("text") or r.get("review"),
        })
    return reviews


all_reviews = []

for url in prof_links:
    prof_id = extract_prof_id(url)
    prof = fetch_professor(prof_id)
    if not prof:
        continue
    prof_reviews = extract_reviews_from_prof(prof, url)
    print(f"{url}: extracted {len(prof_reviews)} reviews")
    all_reviews.extend(prof_reviews)

print(f"Total reviews across all professors: {len(all_reviews)}")


output_json = Path("rmp_reviews_raw.json")
output_json.write_text(
    json.dumps(all_reviews, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"Saved {len(all_reviews)} reviews to {output_json}")