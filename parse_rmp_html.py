from pathlib import Path
from bs4 import BeautifulSoup
import json

HTML_DIR = Path("html_files")

PROF_META = {
    # filename as it appears in html_files/
    "Joseph-Kamto-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Joseph Kamto",
        "school_name": "University of Houston Downtown",
    },
    "Ling-Xu-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Ling Xu",
        "school_name": "University of Houston Downtown",
    },
    "Cyril-Harris-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Cyril Harris",
        "school_name": "University of Houston Downtown",
    },
    "Azadeh-Izadi-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Azadeh Izadi",
        "school_name": "University of Houston Downtown",
    },
    "Emre-Yilmaz-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Emre Yilmaz",
        "school_name": "University of Houston Downtown",
    },
    "Hong-Lin-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Hong Lin",
        "school_name": "University of Houston Downtown",
    },
    "Shengli-Yuan-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Shengli Yuan",
        "school_name": "University of Houston Downtown",
    },
    "Subash-Pakhrin-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Subash Pakhrin",
        "school_name": "University of Houston Downtown",
    },
    "Ting-Zhang-at-University-of-Houston-Downtown-_-Rate-My-Professors.html.html": {
        "professor_name": "Ting Zhang",
        "school_name": "University of Houston Downtown",
    },
    "University-of-Houston-Downtown-CS-Reddit.html.html": {
        "professor_name": None,
        "school_name": "University of Houston Downtown",
    },
}

# automatically pick up every .html file in html_files/
HTML_FILES = list(HTML_DIR.glob("*.html"))


def parse_rmp_file(path: Path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # debug: check whether the raw HTML even has these strings
    print(f"Debug for {path.name}:")
    print("  has Rating__StyledRating prefix?", "Rating__StyledRating-" in html)
    print("  has Comments__StyledComments prefix?", "Comments__StyledComments-" in html)

    meta = PROF_META.get(path.name, {})
    professor_name = meta.get("professor_name")
    school_name = meta.get("school_name")

    reviews = []

    # all review “cards”
    review_cards = soup.select("div.Rating__StyledRating-sc-1rhvpxz-1")

    for card in review_cards:
        comment_el = card.select_one("div.Comments__StyledComments-dzzyvm-0")
        course_el = card.select_one("div.RatingHeader__StyledClass-sc-1dlkqw1-3")
        date_el = card.select_one("div.TimeStamp__StyledTimeStamp-sc-9q2r30-0")

        # Quality and Difficulty numbers
        rating_nums = card.select("div.CardNumRating__CardNumRatingNumber-sc-17t4b9u-2")
        rating_overall = None
        difficulty = None
        if len(rating_nums) > 0:
            txt = rating_nums[0].get_text(strip=True)
            rating_overall = float(txt) if txt else None
        if len(rating_nums) > 1:
            txt = rating_nums[1].get_text(strip=True)
            difficulty = float(txt) if txt else None

        # Helpful up/down counts
        # There are usually two of these per card: first = thumbs up, second = thumbs down.[file:158]
        helpful_els = card.select("div.ThumbsHelpTotalNumber-sc-19shlav-2")
        helpful_up = None
        helpful_down = None
        if len(helpful_els) > 0:
            txt = helpful_els[0].get_text(strip=True)
            helpful_up = int(txt) if txt.isdigit() else 0
        if len(helpful_els) > 1:
            txt = helpful_els[1].get_text(strip=True)
            helpful_down = int(txt) if txt.isdigit() else 0

        review_text = comment_el.get_text(" ", strip=True) if comment_el else None
        course = course_el.get_text(" ", strip=True) if course_el else None
        date = date_el.get_text(" ", strip=True) if date_el else None

        review_data = {
            "professor_name": professor_name,
            "professor_id": None,
            "school_name": school_name,
            "school_id": None,
            "source_url": None,
            "course": course,
            "rating_overall": rating_overall,
            "difficulty": difficulty,
            "date": date,
            "review_text": review_text,
        }

        # Only include helpful counts if there is at least one like or dislike
        if (helpful_up is not None and helpful_up > 0) or (helpful_down is not None and helpful_down > 0):
            review_data["helpful_up"] = helpful_up
            review_data["helpful_down"] = helpful_down

        reviews.append(review_data)

    return reviews


def main():
    all_reviews = []

    for path in HTML_FILES:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        print(f"Parsing {path} ...")
        file_reviews = parse_rmp_file(path)
        print(f"  Found {len(file_reviews)} reviews")
        all_reviews.extend(file_reviews)

    output_json = Path("rmp_reviews_raw.json")
    output_json.write_text(
        json.dumps(all_reviews, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(all_reviews)} reviews to {output_json.resolve()}")


if __name__ == "__main__":
    main()