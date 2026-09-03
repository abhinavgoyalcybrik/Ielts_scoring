# Generic answer-profile transformations (writing_eval_harness coverage
# expansion). Given ANY profile-1 (strong, error-free) base essay from
# profile1_essays.py, these functions mechanically derive the other 12
# profiles at run time - this is deliberately NOT 30 questions x 13
# hand-written essays (390 essays isn't tractable to hand-author and verify
# at the rigor this suite requires). Instead:
#
# - Structural profiles (paragraphing, length, off-topic) are fully
#   generic and essay-content-independent - the same techniques already
#   proven correct in degradation_ladder.py and perturbation.py, just
#   applied to an arbitrary base text instead of one hardcoded example.
# - Vocabulary/grammar profiles use a curated, broadly-applicable
#   substitution dictionary (common academic-essay words -> simpler
#   equivalents) rather than per-essay hand-specified injections. This is
#   the same "verifiable by construction" principle as error_injection.py,
#   generalized: every substitution pair is a real, defensible
#   simplification, just matched by regex across whichever essays happen
#   to contain that word, rather than hand-targeted at one exact phrase.
#   Coverage on any single essay is therefore best-effort (an essay using
#   none of the dictionary's words yields a smaller effect) - this is
#   reported, not hidden (see coverage_matrix.py's per-cell substitution
#   count).

import re


# ---------------------------------------------------------------------------
# Structural transforms - fully generic, no per-essay customisation needed.
# ---------------------------------------------------------------------------
def strip_paragraphing(text: str) -> str:
    """Profile 10: single block of text, no paragraph breaks."""
    return " ".join(p.strip() for p in text.split("\n\n") if p.strip())


def _split_sentences_simple(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def over_paragraph(text: str) -> str:
    """Profile 11: one sentence per paragraph - the opposite structural
    extreme from profile 10."""
    flat = strip_paragraphing(text)
    sentences = _split_sentences_simple(flat)
    return "\n\n".join(sentences)


def truncate_underlength(text: str, target_words: int) -> str:
    """Profile 9: cut the essay down to below the task's minimum word
    count (150 for Task 1, 250 for Task 2) - truncates at a sentence
    boundary where possible, same technique as
    evaluators/writing.py's own _truncate_to_sentence_boundary()."""
    words = text.split()
    if len(words) <= target_words:
        return text
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    kept_paragraphs = []
    word_count = 0
    for para in paragraphs:
        para_words = para.split()
        if word_count >= target_words:
            break
        kept_paragraphs.append(para)
        word_count += len(para_words)
    result = "\n\n".join(kept_paragraphs)
    # Hard cap even within the last kept paragraph, so this is genuinely
    # underlength rather than just "close to the limit".
    result_words = result.split()
    if len(result_words) > target_words:
        result = " ".join(result_words[:target_words])
    return result


_OFF_TOPIC_PARAGRAPH = (
    "Coastal cities around the world experience highly variable weather "
    "patterns depending on ocean currents and seasonal wind direction. In "
    "the winter months, cold currents can lower average temperatures by "
    "several degrees, while in summer, warm currents often bring humid "
    "air that increases the likelihood of afternoon storms. Local "
    "governments in these regions frequently invest in flood defences to "
    "manage the resulting rainfall."
)

_OFF_TOPIC_FULL_ESSAY = (
    "Migratory birds travel remarkable distances each year, often "
    "navigating using a combination of the earth's magnetic field, the "
    "position of the sun, and familiar landmarks passed down through "
    "generations.\n\n"
    "Many species undertake these journeys twice annually, moving toward "
    "warmer regions before winter and returning to their breeding grounds "
    "in spring. The physical demands of migration are considerable, and "
    "birds typically build up substantial fat reserves before departure "
    "to sustain them through the journey.\n\n"
    "Climate change has begun to disrupt these long-established "
    "patterns, with some species now migrating earlier or later than "
    "historical norms, sometimes arriving at breeding grounds before "
    "their usual food sources have become available."
)


def make_partially_off_topic(text: str) -> str:
    """Profile 8: one body paragraph replaced with unrelated content - the
    same technique as degradation_ladder.py's level 4, generalised to
    target whichever paragraph is structurally a body paragraph (not the
    first or last, which are more likely to be intro/conclusion)."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        # Too short to have a clear body paragraph distinct from
        # intro/conclusion - replace the middle-most paragraph anyway,
        # best effort.
        target_index = len(paragraphs) // 2
    else:
        target_index = len(paragraphs) // 2
    paragraphs[target_index] = _OFF_TOPIC_PARAGRAPH
    return "\n\n".join(paragraphs)


def make_fully_off_topic(text: str) -> str:
    """Profile 7: answers a different question entirely. Word count
    roughly matched to the original so this isn't confounded with the
    underlength profile."""
    target_words = len(text.split())
    filler = _OFF_TOPIC_FULL_ESSAY
    while len(filler.split()) < target_words:
        filler += "\n\n" + _OFF_TOPIC_FULL_ESSAY
    words = filler.split()
    return " ".join(words[:target_words]) if len(words) > target_words else filler


# ---------------------------------------------------------------------------
# Vocabulary transform - generic dictionary substitution.
# ---------------------------------------------------------------------------
# Sophisticated/precise word -> simpler/vaguer equivalent. Deliberately
# common, general-academic-register words likely to appear across many
# different essays on different topics, not phrases specific to any one
# essay's subject matter.
_VOCAB_SIMPLIFY_MAP = {
    "significant": "big", "significantly": "a lot",
    "considerable": "big", "considerably": "a lot",
    "substantial": "big", "substantially": "a lot",
    "consequently": "so", "furthermore": "also", "moreover": "also",
    "nevertheless": "but", "nonetheless": "but", "however": "but",
    "individuals": "people", "numerous": "many", "approximately": "about",
    "demonstrate": "show", "demonstrates": "shows",
    "utilise": "use", "utilize": "use", "utilised": "used", "utilized": "used",
    "predominantly": "mostly", "acquire": "get", "acquired": "got",
    "beneficial": "good", "detrimental": "bad", "advantageous": "good",
    "subsequently": "after", "ultimately": "finally",
    "obtain": "get", "obtained": "got",
    "comprehensive": "full", "particularly": "specially",
    "essential": "needed", "essentially": "basically",
    "sufficient": "enough", "insufficient": "not enough",
    "regarding": "about", "concerning": "about",
    "facilitate": "help", "facilitates": "helps",
    "contribute": "add", "contributes": "adds", "contributed": "added",
    "perspective": "view", "perspectives": "views",
    "circumstances": "situation",
    "reluctant": "not wanting", "inevitable": "sure to happen",
    "primarily": "mainly", "alternative": "other", "alternatives": "other options",
    "sophisticated": "smart", "diminish": "make smaller",
    "enhance": "make better", "enhances": "makes better",
    "implement": "do", "implemented": "did",
    "adverse": "bad", "adversely": "badly",
    "genuinely": "really",
    # Common connectors and reporting verbs - deliberately everyday words
    # likely to appear in almost any essay, since the first version's
    # dictionary (rarer, more "sophisticated" words only) had genuinely
    # sparse coverage - measured at an average of 1.6 matches per essay,
    # far too few to meaningfully weaken Lexical Resource.
    "important": "big", "difficult": "hard", "increase": "go up",
    "increased": "went up", "increases": "goes up", "increasing": "going up",
    "decrease": "go down", "decreased": "went down", "decreases": "goes down",
    "because": "cause of", "although": "but", "therefore": "so",
    "indicates": "shows", "indicate": "show", "reason": "cause",
    "argue": "say", "argues": "says", "believe": "think", "believes": "thinks",
    "understand": "get", "achieve": "get", "achieved": "got",
    "improve": "make better", "improved": "made better", "improves": "makes better",
    "reduce": "make less", "reduced": "made less", "reduces": "makes less",
    "develop": "grow", "developed": "grew", "develops": "grows",
    "create": "make", "created": "made", "creates": "makes",
    "produce": "make", "produced": "made", "produces": "makes",
    "include": "have", "includes": "has", "included": "had",
    "involve": "have", "involves": "has", "involved": "had",
    "specific": "exact", "particular": "certain", "various": "different",
    "several": "some", "majority": "most", "minority": "few",
    "extremely": "very", "highly": "very", "relatively": "kind of",
    "generally": "usually", "typically": "usually", "commonly": "usually",
    "frequently": "often", "rarely": "not often", "occasionally": "sometimes",
    "immediately": "right away", "eventually": "in the end",
    "gradually": "slowly", "rapidly": "fast", "dramatically": "a lot",
    "slightly": "a little", "extensive": "big", "widespread": "common",
    "adequate": "enough", "appropriate": "right", "effective": "good",
    "efficient": "good", "convenient": "easy", "sustainable": "lasting",
    "innovative": "new", "traditional": "old", "conventional": "normal",
    "contemporary": "modern", "fundamental": "basic", "crucial": "important",
    "vital": "important", "critical": "important",
    "optional": "not needed", "mandatory": "needed", "compulsory": "needed",
    "provide": "give", "provides": "gives", "provided": "gave",
    "require": "need", "requires": "needs", "required": "needed",
    "maintain": "keep", "maintains": "keeps", "maintained": "kept",
    "encourage": "help", "encourages": "helps", "encouraged": "helped",
    "prevent": "stop", "prevents": "stops", "prevented": "stopped",
    "address": "deal with", "addresses": "deals with", "addressed": "dealt with",
}


def simplify_vocabulary(text: str, intensity: float = 1.0) -> tuple:
    """Replaces dictionary words with simpler equivalents. intensity in
    [0, 1] controls what fraction of the dictionary's matches in this
    specific text get replaced (deterministic - always the first N
    matches in reading order, not random, so results are reproducible).
    Returns (new_text, n_substitutions_made) - the count matters for
    reporting how much effect this transform actually had on a given
    essay (see module docstring on best-effort coverage)."""
    matches = []
    for sophisticated, simple in _VOCAB_SIMPLIFY_MAP.items():
        for m in re.finditer(rf"\b{re.escape(sophisticated)}\b", text, flags=re.IGNORECASE):
            matches.append((m.start(), m.end(), simple, m.group(0)))
    matches.sort(key=lambda t: t[0])
    n_to_apply = max(0, round(len(matches) * intensity)) if matches else 0
    applied = matches[:n_to_apply]
    applied.sort(key=lambda t: t[0], reverse=True)  # apply back-to-front so offsets stay valid
    result = text
    for start, end, simple, original in applied:
        replacement = simple.capitalize() if original[0].isupper() else simple
        result = result[:start] + replacement + result[end:]
    return result, len(applied)


# ---------------------------------------------------------------------------
# Grammar/structure degradation - inserted broken sentences (guaranteed
# broken by construction, since they're new text, not a mutation of
# existing grammar that might accidentally stay correct) + generic
# complex-to-simple sentence splitting.
# ---------------------------------------------------------------------------
_BROKEN_TEMPLATE_SENTENCES = [
    "There is many reason for this situation.",
    "This problem it happen because of many factor.",
    "In my opinion I think that this issue is very important thing.",
    "Many people they believe this is true.",
]


def _simplify_sentence_structure(text: str, intensity: float = 1.0) -> str:
    """Splits sentences joined by 'although'/'while'/'because' into two
    separate simple sentences, reducing grammatical range without
    injecting a specific wrong-vs-right error - a genuine range
    reduction, not a fabricated mistake, matching how GRA Band 5-6 is
    actually described (limited/repetitive structures) rather than
    inventing ungrammatical text."""
    connectors = [
        (r",\s+although\s+", ". However, "),
        (r",\s+while\s+", ". Meanwhile, "),
        (r",\s+because\s+", ". This is because "),
        (r",\s+which\s+", ". This "),
    ]
    matches = []
    for pattern, _ in connectors:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((m.start(), m.end(), pattern))
    matches.sort(key=lambda t: t[0])
    n_to_apply = max(0, round(len(matches) * intensity))
    applied = sorted(matches[:n_to_apply], key=lambda t: t[0], reverse=True)
    result = text
    for start, end, pattern in applied:
        for pat, repl in connectors:
            if pat == pattern:
                result = result[:start] + repl + result[end:]
                break
    return result


def inject_grammar_weakness(text: str, n_broken_sentences: int = 3, split_intensity: float = 0.6) -> str:
    """Profiles 3 and 5: reduces grammatical range/accuracy generically.
    Two mechanisms, both safe to apply to arbitrary text: (a) splits some
    complex sentences into simpler ones (range reduction, not an
    error), and (b) appends a small number of guaranteed-broken template
    sentences to the end of the essay's body (genuine errors, but newly
    inserted text rather than a mutation of existing correct grammar, so
    there's no risk of the "error" accidentally remaining grammatical)."""
    result = _simplify_sentence_structure(text, intensity=split_intensity)
    paragraphs = [p for p in result.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        insert_at = len(paragraphs) // 2
        broken = " ".join(_BROKEN_TEMPLATE_SENTENCES[:n_broken_sentences])
        paragraphs[insert_at] = paragraphs[insert_at] + " " + broken
        result = "\n\n".join(paragraphs)
    else:
        result = result + " " + " ".join(_BROKEN_TEMPLATE_SENTENCES[:n_broken_sentences])
    return result


# ---------------------------------------------------------------------------
# L1-interference patterns (profile 13) - Punjabi/Hindi-typical transfer
# errors, each a targeted, generic regex applied wherever a matching
# context exists in the given essay (best-effort - not every essay will
# contain every pattern's trigger context).
# ---------------------------------------------------------------------------
def inject_l1_patterns(text: str) -> tuple:
    """Returns (new_text, patterns_applied) - patterns_applied lists which
    L1-transfer patterns actually found a match in this specific essay,
    since coverage is best-effort per essay (see module docstring).

    First version's 4 patterns were narrow enough (a fixed 4-noun list,
    "for N years" only, "was" only) that 13/30 essays matched nothing at
    all - measured directly, not assumed. Broadened each pattern's
    trigger set and added a tense-inconsistency direction that works on
    present-tense essays too, so genuine coverage gaps (an essay that
    truly contains none of these trigger words) are now the exception,
    not close to half the corpus."""
    applied = []
    result = text

    # Preposition transfer: "for N years/months/days" -> "since N ..."
    new_result, n = re.subn(
        r"\bfor\s+(\d+|\w+)\s+(years?|months?|days?|weeks?)\b",
        r"since \1 \2", result, count=1, flags=re.IGNORECASE,
    )
    if n:
        result = new_result
        applied.append("preposition_transfer_since_for")

    # Article omission before a common abstract/mass noun - a much larger
    # candidate list than the original 4, covering common IELTS-topic
    # nouns.
    _ARTICLE_OMISSION_NOUNS = [
        "education", "technology", "government", "society", "environment",
        "economy", "health", "healthcare", "employment", "unemployment",
        "poverty", "crime", "pollution", "transport", "transportation",
        "communication", "information", "knowledge", "experience",
        "internet", "media", "industry", "agriculture", "tourism",
    ]
    for noun in _ARTICLE_OMISSION_NOUNS:
        new_result, n = re.subn(rf"\bthe {noun}\b", noun, result, count=1, flags=re.IGNORECASE)
        if n:
            result = new_result
            applied.append(f"article_omission_{noun}")
            break

    # Tense inconsistency, either direction: a past-tense essay gets one
    # "was" flipped to "is"; a present-tense essay (the more common case
    # for Task 2 opinion/discussion essays, which rarely use "was" at
    # all) gets one "is" flipped to "was" instead - the reverse
    # direction, equally genuine L1 tense-mapping interference, and
    # reliably present since almost every essay uses "is" somewhere.
    new_result, n = re.subn(r"\bwas\b", "is", result, count=1)
    if n:
        result = new_result
        applied.append("tense_inconsistency_was_to_is")
    else:
        new_result, n = re.subn(r"\bis\b", "was", result, count=1)
        if n:
            result = new_result
            applied.append("tense_inconsistency_is_to_was")

    # Direct-translation collocation: "very much" appended after a common
    # intensifiable adjective - a well-documented Hindi/Punjabi-speaker
    # transfer pattern. Expanded adjective list for broader coverage.
    _INTENSIFIABLE_ADJECTIVES = (
        r"important|difficult|useful|interesting|common|popular|effective|"
        r"necessary|helpful|dangerous|expensive|convenient|beneficial|"
        r"harmful|serious|obvious"
    )
    new_result, n = re.subn(
        rf"\b({_INTENSIFIABLE_ADJECTIVES})\b(?!\s+very much)",
        r"\1 very much",
        result, count=1, flags=re.IGNORECASE,
    )
    if n:
        result = new_result
        applied.append("direct_translation_very_much")

    return result, applied


# ---------------------------------------------------------------------------
# Memorised/template answer (profile 6) - a genuine formulaic essay FRAME
# (generic opener, generic signpost transitions, generic shallow reasoning
# in the body, generic conclusion) with the question's topic phrase bolted
# on at a handful of points - this is what a memorised template actually
# looks like in real candidate submissions: a fixed skeleton, reused across
# many different questions, with the topic word swapped in.
#
# Earlier version of this function instead vocabulary-flattened the
# *entire* base essay (via simplify_vocabulary) and prepended one generic
# opener sentence to it. That kept 100% of the original essay's specific,
# well-developed reasoning and examples intact - it mostly re-tested
# vocabulary weakness, already covered by p3/p4, rather than testing
# whether Task Response/Achievement penalises generic, formulaic reasoning
# independent of grammar/vocabulary quality, which is the entire point of
# this profile existing separately from p3/p4 (same "does the criterion
# react independently" purpose as p4/p5's grammar/vocab split). Caught
# before ever running this profile against the live API - see the module
# docstring's coverage-measurement discipline.
#
# Every {topic} placeholder is used as the object of a preposition ("about
# {topic}", "connected to {topic}") or of "the issue of"/"the topic of",
# never as a bare grammatical subject - this means a topic phrase's
# internal singular/plural form never has to agree with a frame verb, so
# question_bank.py's 30 topic phrases don't need special-casing here.
#
# The Task 1 Academic frame is deliberately the SAME fixed "report on
# change over time" skeleton regardless of chart_type (line graph, pie
# chart, process diagram, map, ...), and the Task 1 GT frame is
# deliberately the same fixed formal-letter skeleton regardless of
# letter_register (formal/semi-formal/informal). This is intentional, not
# an oversight: a memorised template bolted onto a task type/register it
# doesn't actually fit (a "trends over time" frame on a process diagram
# question, or a formal "Dear Sir/Madam... Yours faithfully" frame on an
# informal letter to a friend) is itself a well-documented, realistic Task
# Achievement failure mode - inappropriate format/register for the task -
# not a bug in this profile.
# ---------------------------------------------------------------------------
_TASK2_TEMPLATE_FRAME = [
    "Nowadays, in this modern era, the issue of {topic} is widely "
    "discussed by people all over the world. There are many people who "
    "have different opinions about this in today's society. This essay "
    "will discuss the topic of {topic} and give my own opinion on the "
    "matter before reaching a final conclusion.",

    "Firstly, it is important to think carefully about {topic}, because "
    "it has a significant impact on society as a whole. Many experts "
    "believe that this issue affects people in different ways, and there "
    "are several reasons why this matter deserves careful attention. In "
    "addition, concern about {topic} has grown considerably in recent "
    "years, which clearly shows that more attention should be given to "
    "it by both individuals and the wider community.",

    "On the other hand, some people may hold a different view on this "
    "matter. However, in my personal opinion, the advantages connected "
    "to {topic} outweigh the disadvantages when all factors are taken "
    "into consideration. There are many aspects that need to be examined "
    "before reaching a final conclusion, and it is clear that this is a "
    "complex issue with no single simple answer. It is also worth "
    "remembering that different people in different situations may "
    "reasonably reach different conclusions about this same matter.",

    "In conclusion, from what has been discussed above, I firmly believe "
    "that careful consideration should be given to {topic}, since this "
    "is an important issue for everyone involved. Governments, "
    "organisations, and individuals should work together to address it "
    "effectively, so that a positive outcome can be achieved for society "
    "as a whole in the future.",
]

_TASK1_ACADEMIC_TEMPLATE_FRAME = [
    "The chart illustrates data connected to {topic} over the given "
    "period. Overall, it can be seen that there were some significant "
    "changes during this time, and several clear trends can be "
    "identified from the information provided in the chart.",

    "Looking at the data in more detail, it is clear that there is an "
    "interesting overall pattern regarding {topic}. Several categories "
    "can be observed in the chart, and each one shows a somewhat "
    "different trend over the period shown. This is an important point "
    "to note when analysing the information given in detail.",

    "In addition, other aspects relating to {topic} should also be "
    "taken into account. There were some notable differences between "
    "the various categories shown in the chart, which is a key feature "
    "of this particular data set worth mentioning here.",

    "To conclude, the chart shows several important trends connected to "
    "{topic}, and these changes are significant when considering the "
    "overall pattern shown in the data provided above.",
]

_TASK1_GT_TEMPLATE_FRAME = [
    "Dear Sir/Madam,\n\nI am writing to inform you that I would like to "
    "raise a matter connected to {topic}. I hope this letter finds you "
    "well and I appreciate you taking the time to read it carefully.",

    "Firstly, I would like to mention that this is a very important "
    "matter that needs your urgent attention as soon as possible. There "
    "are several specific reasons why I am writing to you concerning "
    "{topic}, and I would like to explain each of these reasons in the "
    "following paragraphs of this letter.",

    "Furthermore, I believe that appropriate consideration should be "
    "given to {topic} by all parties involved in this situation. It "
    "would be greatly appreciated if this particular matter could be "
    "resolved as soon as possible, at your earliest convenience.",

    "I look forward to hearing from you soon regarding this matter. "
    "Thank you very much for your time and consideration.\n\nYours "
    "faithfully,\nCandidate",
]

_GENERIC_TOPIC_FALLBACK = "this issue"


def make_template_answer(task_type: str, t1_variant: str = None, topic: str = None) -> str:
    """Builds a genuine memorised-template answer: a fixed, generic essay
    frame (see module comment above) with the question's topic phrase
    bolted on at each {topic} placeholder. task_type/t1_variant select
    which frame (Task 1 Academic report / Task 1 GT letter / Task 2
    essay); topic falls back to a generic phrase if not supplied so this
    still returns a usable (if less realistic) template rather than
    raising."""
    topic_phrase = topic or _GENERIC_TOPIC_FALLBACK
    if task_type == "task_1":
        frame = _TASK1_GT_TEMPLATE_FRAME if t1_variant == "general" else _TASK1_ACADEMIC_TEMPLATE_FRAME
    else:
        frame = _TASK2_TEMPLATE_FRAME
    paragraphs = [p.format(topic=topic_phrase) for p in frame]
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Task 1 only: data misread (profile 12) - alters the first number/
# percentage mentioned in the essay to a materially different, incorrect
# value, so the essay's stated data no longer matches the (unstated, in
# this text-only harness) source chart. Generic and reliable since it's
# pure regex numeric substitution on a text already known to be accurate.
# ---------------------------------------------------------------------------
_NUMBER_PATTERN = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*|\d{4})\s*(%|percent|years?|mm|kg|km|thousand|million)?\b",
    re.IGNORECASE,
)

# Spelled-out number words, in the order several of the corpus essays
# actually use them - a fallback pass for when an essay contains no
# digit-form numbers at all (several Task 1 essays in this corpus were
# written with numbers in words, e.g. "thirty percent", not "30%"). Each
# word maps to a DIFFERENT word from this same set as its "altered" value
# - a direct word swap, not arithmetic on parsed values, so it stays
# simple and unambiguous rather than needing to parse compound number
# words like "eighty-five" into an integer.
_WORD_NUMBER_SWAP = {
    "one": "eleven", "two": "twelve", "three": "thirteen", "four": "fourteen",
    "five": "fifteen", "six": "sixteen", "seven": "seventeen", "eight": "eighteen",
    "nine": "nineteen", "ten": "twenty",
    "eleven": "one", "twelve": "two", "thirteen": "three", "fourteen": "four",
    "fifteen": "five", "sixteen": "six", "seventeen": "seven", "eighteen": "eight",
    "nineteen": "nine", "twenty": "ten",
    "thirty": "seventy", "forty": "eighty", "fifty": "ninety",
    "sixty": "twenty", "seventy": "thirty", "eighty": "forty", "ninety": "fifty",
    "half": "a third", "quarter": "two-thirds", "third": "quarter",
}
_WORD_NUMBER_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMBER_SWAP, key=len, reverse=True)) + r")(-\w+)?\b",
    re.IGNORECASE,
)


def misread_data(text: str) -> tuple:
    """Returns (new_text, changed) where changed is a (original, altered)
    tuple for the first numeric data point found, or (text, None) if the
    essay contains none.

    First version of this only matched "%"/"percent" with a 1-3 digit cap
    and silently missed virtually every other numeric format Task 1
    essays actually use - caught when 17/20 Task 1 essays came back "no
    numeric data to alter", which is implausible for a task whose entire
    content is describing chart data. Investigating the specific failures
    found two distinct causes, both fixed here: (a) 4-digit years like
    "2000"/"2020" were excluded by the 1-3 digit cap, and (b) several
    essays spell numbers out in words ("thirty percent") rather than
    digits, which a digit-only regex can never match regardless of cap
    size - these get a dedicated word-number fallback pass. Genuine "no
    data" cases remain for essay types where that's actually correct
    (process diagrams, maps, and General Training letters typically
    describe steps/features/requests rather than quantities - the
    caller/report should treat that as expected, not a gap)."""
    m = _NUMBER_PATTERN.search(text)
    if m:
        original_str = m.group(0)
        digits_str = m.group(1)
        original_num = int(digits_str.replace(",", ""))
        altered_num = original_num + 25 if original_num <= 70 else max(1, original_num - 25)
        altered_digits = f"{altered_num:,}" if "," in digits_str else str(altered_num)
        altered_str = original_str.replace(digits_str, altered_digits, 1)
        new_text = text[: m.start()] + altered_str + text[m.end():]
        return new_text, (original_str, altered_str)

    wm = _WORD_NUMBER_PATTERN.search(text)
    if wm:
        original_str = wm.group(0)
        word = wm.group(1).lower()
        altered_word = _WORD_NUMBER_SWAP[word]
        altered_str = altered_word + (wm.group(2) or "")
        new_text = text[: wm.start()] + altered_str + text[wm.end():]
        return new_text, (original_str, altered_str)

    return text, None


# ---------------------------------------------------------------------------
# Profile registry - name -> (description, generator(text) -> str-or-tuple)
# Central place the coverage matrix iterates over.
# ---------------------------------------------------------------------------
PROFILE_NAMES = [
    "p1_strong",
    "p2_competent_occasional_errors",
    "p3_weak",
    "p4_strong_grammar_weak_vocab",
    "p5_weak_grammar_strong_vocab",
    "p6_memorised_template",
    "p7_off_topic",
    "p8_partially_off_topic",
    "p9_underlength",
    "p10_no_paragraphing",
    "p11_over_paragraphed",
    "p12_misreads_data",  # Task 1 only
    "p13_l1_influenced",
]


def build_profile(profile_name: str, base_text: str, task_type: str, t1_variant: str = None, topic: str = None) -> dict:
    """Given a profile-1 base essay, returns {text, note} for the
    requested profile. "note" carries any per-call metadata a check might
    need (e.g. how many substitutions were actually made, for reporting
    coverage honestly rather than assuming every transform hit every
    essay equally). t1_variant/topic are only used by
    p6_memorised_template (see make_template_answer's module comment for
    why it needs the real question topic rather than deriving anything
    from base_text)."""
    if profile_name == "p1_strong":
        return {"text": base_text, "note": None}
    if profile_name == "p2_competent_occasional_errors":
        text, n = simplify_vocabulary(base_text, intensity=0.2)
        return {"text": text, "note": f"{n} vocab substitutions"}
    if profile_name == "p3_weak":
        text, n = simplify_vocabulary(base_text, intensity=0.7)
        text = inject_grammar_weakness(text, n_broken_sentences=3, split_intensity=0.8)
        return {"text": text, "note": f"{n} vocab substitutions + grammar weakness"}
    if profile_name == "p4_strong_grammar_weak_vocab":
        text, n = simplify_vocabulary(base_text, intensity=1.0)
        return {"text": text, "note": f"{n} vocab substitutions, grammar untouched"}
    if profile_name == "p5_weak_grammar_strong_vocab":
        text = inject_grammar_weakness(base_text, n_broken_sentences=4, split_intensity=1.0)
        return {"text": text, "note": "grammar weakened, vocab untouched"}
    if profile_name == "p6_memorised_template":
        text = make_template_answer(task_type, t1_variant=t1_variant, topic=topic)
        return {"text": text, "note": f"template frame, topic={topic!r}"}
    if profile_name == "p7_off_topic":
        return {"text": make_fully_off_topic(base_text), "note": None}
    if profile_name == "p8_partially_off_topic":
        return {"text": make_partially_off_topic(base_text), "note": None}
    if profile_name == "p9_underlength":
        target = 100 if task_type == "task_1" else 180
        return {"text": truncate_underlength(base_text, target), "note": f"target {target} words"}
    if profile_name == "p10_no_paragraphing":
        return {"text": strip_paragraphing(base_text), "note": None}
    if profile_name == "p11_over_paragraphed":
        return {"text": over_paragraph(base_text), "note": None}
    if profile_name == "p12_misreads_data":
        if task_type != "task_1":
            return {"text": None, "note": "not applicable outside Task 1"}
        text, changed = misread_data(base_text)
        return {"text": text, "note": f"changed {changed}" if changed else "no numeric data found to alter"}
    if profile_name == "p13_l1_influenced":
        text, applied = inject_l1_patterns(base_text)
        return {"text": text, "note": f"patterns applied: {applied}" if applied else "no matching context found"}
    raise ValueError(f"Unknown profile: {profile_name}")
