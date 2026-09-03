# Clean corpus (Speaking eval harness) - self-authored, grammatically
# error-free Part 1/2/3 question+answer transcripts, mirroring
# tests/writing_eval_harness/clean_corpus.py's role: expect (near) zero
# mistakes when run through generate_question_mistakes()/generate_scores().
# Every transcript is original text written for this test suite, not
# copied from any IELTS/Cambridge publication - same discipline as the
# Writing harness's corpus.
#
# Kept deliberately "written-clean" (normal punctuation, no fillers/self-
# corrections) rather than realistic messy ASR output - that's what
# asr_artifact_corpus.py is for. This corpus isolates "does the pipeline
# invent errors in genuinely correct English", the ASR corpus isolates
# "does the pipeline correctly ignore ASR-only artifacts and normal
# spoken-language features".

CLEAN_PART1 = [
    {
        "id": "p1_hometown",
        "part": 1,
        "question": "Can you tell me about your hometown?",
        "answer": (
            "I come from a mid-sized city on the coast. It's known for its "
            "old fishing harbour and a long promenade where people go "
            "walking in the evenings. What I like most about it is that "
            "it's small enough to get around easily but still has good "
            "restaurants and a couple of decent parks."
        ),
    },
    {
        "id": "p1_work_study",
        "part": 1,
        "question": "Do you work or are you a student?",
        "answer": (
            "I'm currently working as a junior accountant at a small "
            "firm. I've been there for about a year now. It's my first "
            "proper job after finishing my degree, and I'm still learning "
            "a lot, but I enjoy the work because every client's situation "
            "is a bit different."
        ),
    },
    {
        "id": "p1_hobbies",
        "part": 1,
        "question": "What do you like to do in your free time?",
        "answer": (
            "I spend a lot of my free time cycling. There's a river path "
            "near my house that I use most weekends, and occasionally I "
            "join a small group for longer rides. Apart from that I read "
            "quite a bit, mostly history books."
        ),
    },
]

CLEAN_PART2 = {
    "id": "p2_memorable_trip",
    "part": 2,
    "question": (
        "Describe a memorable trip you have taken. You should say: where "
        "you went, who you went with, what you did there, and explain why "
        "it was memorable."
    ),
    "answer": (
        "I'd like to talk about a trip I took to the mountains with two "
        "close friends about three years ago. We had been planning it for "
        "months, ever since one of my friends suggested we try hiking "
        "somewhere none of us had been before, so we picked a national "
        "park a few hours from the city.\n\n"
        "We spent four days there in total. Most of our time was taken up "
        "with hiking, but we also stayed in a small wooden cabin with no "
        "internet connection at all, which felt strange at first but "
        "turned out to be one of the best parts of the trip. In the "
        "evenings we cooked simple meals together and talked for hours "
        "instead of looking at our phones.\n\n"
        "The trip was memorable for a few reasons. Firstly, the scenery "
        "was more dramatic than anything I had seen before, especially on "
        "the second day when we reached a ridge with views over the whole "
        "valley. Secondly, it was the first time the three of us had "
        "travelled together without any particular plan, and having to "
        "figure things out as we went actually brought us closer. Even "
        "now, we still talk about that trip whenever we meet up, and it's "
        "become something of a running joke between us that we underestimated "
        "how tiring the second day would be."
    ),
}

CLEAN_PART3 = [
    {
        "id": "p3_travel_changes",
        "part": 3,
        "question": "How has tourism changed in your country over the last twenty years?",
        "answer": (
            "I think tourism has grown enormously, mainly because "
            "domestic flights have become much cheaper and more frequent "
            "than they used to be. Places that were once considered "
            "difficult to reach are now marketed heavily to visitors, "
            "which has brought a lot of investment into smaller towns but "
            "has also put pressure on the local infrastructure in some "
            "areas."
        ),
    },
    {
        "id": "p3_technology_travel",
        "part": 3,
        "question": "Do you think technology has made travelling easier or more complicated?",
        "answer": (
            "On balance, I'd say easier. Booking flights and "
            "accommodation used to require a travel agent, whereas now "
            "you can compare dozens of options in a few minutes. That "
            "said, relying so heavily on apps for navigation and "
            "translation can be a problem when there's no signal, so "
            "travellers sometimes end up less prepared for that "
            "situation than people were in the past."
        ),
    },
    {
        "id": "p3_sustainable_tourism",
        "part": 3,
        "question": "What can governments do to encourage more sustainable tourism?",
        "answer": (
            "Governments could limit the number of visitors allowed into "
            "particularly fragile sites at any one time, and invest the "
            "revenue from tourism directly back into conservation rather "
            "than only into new hotels. Offering incentives to companies "
            "that use environmentally friendly transport would also make "
            "a real difference over time."
        ),
    },
]


def combined_transcript_for_part(part: int) -> str:
    """Builds the "Q: ...\\nA: ..." combined-transcript string
    generate_scores() expects for one part, from all clean-corpus items in
    that part - matches the shape used in generate_scores(part_number,
    combined_transcripts, ...) throughout evaluators/speaking_audio.py."""
    if part == 1:
        items = CLEAN_PART1
    elif part == 2:
        items = [CLEAN_PART2]
    elif part == 3:
        items = CLEAN_PART3
    else:
        raise ValueError(f"Unknown part: {part}")
    return "\n\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in items)


ALL_CLEAN_ITEMS = CLEAN_PART1 + [CLEAN_PART2] + CLEAN_PART3
