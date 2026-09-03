# ASR-artifact corpus (Speaking eval harness) - self-authored transcripts
# written the way real ASR (speech-to-text) output actually looks: no
# punctuation, no capitalization, filler words, one self-correction/false
# start, contractions. The underlying content is grammatically correct
# spoken English - the ONLY thing "wrong" here is exactly what real
# transcription introduces, none of which the candidate actually did.
# Expected mistake count: 0. This is the harness's direct test of the
# guards in evaluators/speaking_audio.py's DO-NOT-FLAG prompt blocks and
# _validate_question_mistakes()'s punctuation/self-correction backstops -
# the one place Speaking genuinely differs from Writing, and the
# confirmed largest share of its false-positive risk.

ASR_ARTIFACT_ITEMS = [
    {
        "id": "asr_hometown",
        "part": 1,
        "question": "Can you tell me about your hometown?",
        "answer": (
            "um so i'm from a mid sized city on the coast it's it's known "
            "for its old fishing harbour and theres a long promenade "
            "where people go walking in the evenings i guess what i like "
            "most about it is that its small enough to get around easily "
            "but it still has good restaurants and a couple of decent "
            "parks"
        ),
    },
    {
        "id": "asr_work_study",
        "part": 1,
        "question": "Do you work or are you a student?",
        "answer": (
            "i'm currently working as a junior accountant at a small firm "
            "uh i've been there for about a year now its my first proper "
            "job after finishing my degree and i'm still learning a lot "
            "but i really enjoy the work because every clients situation "
            "is a bit different"
        ),
    },
    {
        "id": "asr_memorable_trip",
        "part": 2,
        "question": (
            "Describe a memorable trip you have taken. You should say: "
            "where you went, who you went with, what you did there, and "
            "explain why it was memorable."
        ),
        "answer": (
            "id like to talk about a trip i took to the mountains with "
            "two close friends about three years ago we had been "
            "planning it for months ever since one of my friends "
            "suggested we try hiking somewhere none of us had been before "
            "so we picked a national park a few hours from the city we "
            "spent four days there in total most of our time was taken "
            "up with hiking but we also stayed in a small wooden cabin "
            "with no internet connection at all which felt strange at "
            "first but it turned out to be one of the best parts of the "
            "trip"
        ),
    },
    {
        "id": "asr_technology_travel",
        "part": 3,
        "question": "Do you think technology has made travelling easier or more complicated?",
        "answer": (
            "on balance id say easier booking flights and accommodation "
            "used to require a travel agent whereas now you can compare "
            "dozens of options in a few minutes that said relying so "
            "heavily on apps for navigation and translation can be a "
            "problem when theres no signal so travellers sometimes end "
            "up less prepared for that situation than people were in the "
            "past"
        ),
    },
]
