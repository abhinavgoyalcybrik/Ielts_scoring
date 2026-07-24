def _normalize_speech_rate_for_pronunciation(rate: float) -> float:
    """0-1 goodness score for speech rate as a pronunciation/delivery signal.
    Ideal conversational range ~110-160 wpm; falls off toward 0 the further
    outside that range (too slow = labored articulation, too fast = rushed
    and often unclear)."""
    if not rate or rate <= 0:
        return 0.5
    if 110 <= rate <= 160:
        return 1.0
    if rate < 110:
        return max(0.0, rate / 110)
    # rate > 160
    return max(0.0, 1.0 - (rate - 160) / 120)


# Composite (0-1) -> IELTS Pronunciation band, anchored explicitly to the
# official Pronunciation descriptor tiers rather than a bare linear formula.
# Each threshold is the composite level that first satisfies that band's
# actual descriptor language:
#   9: 0.97+  "Full range of phonological features...effortless to
#              understand throughout; accent has NO effect on intelligibility."
#   8: 0.85+  "Wide range of phonological features...sustained, flexible
#              use...easy to understand throughout; accent has MINIMAL
#              effect on intelligibility."
#   7: 0.72+  "Displays all the positive features of band 6 and SOME,
#              but not all, of the positive features of band 8."
#   6: 0.60+  "Range of phonological features but control is variable...
#              can generally be understood, though mispronunciation
#              causes occasional lapses."
#   5: 0.48+  "Control is variable...overall meaning can be unclear in
#              longer utterances."
#   4: 0.35+  "Limited range of features; frequent lapses in control...
#              requires significant listener effort, patches of speech
#              may be unintelligible."
#   3: 0.20+  "Displays some features of band 2, and some, but not all,
#              of the positive features of band 4." (no independent
#              descriptor text of its own - same blend pattern as bands 5/7)
#   2: 0.08+  "Often unintelligible."
#   1: below  "No rateable speech produced."
# Interpolated linearly WITHIN each band so half-bands (6.5, 7.5, etc.)
# remain possible - this isn't a discrete 9-point lookup, the boundaries
# themselves are what's anchored to the descriptor text. Calibrated so a
# genuinely poor reading (weak articulation/stress/intonation together)
# lands in the 2-3 range the descriptors actually describe for that
# quality of evidence, not an overly generous middle band.
_PRONUNCIATION_BAND_ANCHORS = [
    (0.00, 1.0),
    (0.08, 2.0),
    (0.20, 3.0),
    (0.35, 4.0),
    (0.48, 5.0),
    (0.60, 6.0),
    (0.72, 7.0),
    (0.85, 8.0),
    (0.97, 9.0),
    (1.00, 9.0),
]


def _composite_to_pronunciation_band(composite: float) -> float:
    composite = max(0.0, min(1.0, composite))
    for (low_x, low_band), (high_x, high_band) in zip(_PRONUNCIATION_BAND_ANCHORS, _PRONUNCIATION_BAND_ANCHORS[1:]):
        if low_x <= composite <= high_x:
            if high_x == low_x:
                return high_band
            fraction = (composite - low_x) / (high_x - low_x)
            return low_band + fraction * (high_band - low_band)
    return 9.0


def compute_pronunciation_score(acoustic_features: dict, asr_confidence: float = 1.0) -> float:
    """
    Pronunciation score derived ENTIRELY from acoustic evidence - pause
    patterns, stress accuracy, intonation, speech rate, and articulation
    (phoneme accuracy / mispronunciation rate) - never from transcript
    wording or vocabulary choice. Per the IELTS Pronunciation descriptor,
    this criterion is fundamentally about how the response SOUNDS, so text
    content has no bearing on it at all.

    Spans the genuine IELTS 1.0-9.0 band range rather than being
    structurally capped near the middle.
    """
    if not acoustic_features:
        return 5.5

    phoneme_accuracy = acoustic_features.get("phoneme_accuracy", 0.7)
    phoneme_accuracy = 0.7 if phoneme_accuracy is None else phoneme_accuracy

    stress_accuracy = acoustic_features.get("stress_accuracy", 0.7)
    stress_accuracy = 0.7 if stress_accuracy is None else stress_accuracy

    intonation = acoustic_features.get("intonation_score", 0.6)
    intonation = 0.6 if intonation is None else intonation

    audio_quality = acoustic_features.get("audio_quality_score", 6.0)
    audio_quality = 6.0 if audio_quality is None else audio_quality

    mispronunciation_rate = acoustic_features.get("mispronunciation_rate", 0.3)
    mispronunciation_rate = 0.3 if mispronunciation_rate is None else mispronunciation_rate

    pause_count = acoustic_features.get("pause_count", 3)
    pause_count = 3 if pause_count is None else pause_count

    avg_pause = acoustic_features.get("avg_pause_duration", 0.6)
    avg_pause = 0.6 if avg_pause is None else avg_pause

    speech_rate = acoustic_features.get("speech_rate") or acoustic_features.get("speech_rate_wpm") or 0.0

    # Normalize each raw signal onto a 0.0 (worst) - 1.0 (best) scale.
    articulation_norm = max(0.0, min(1.0, (phoneme_accuracy + (1.0 - mispronunciation_rate)) / 2))
    stress_norm = max(0.0, min(1.0, stress_accuracy))
    intonation_norm = max(0.0, min(1.0, (intonation - 0.3) / 0.5))
    speech_rate_norm = _normalize_speech_rate_for_pronunciation(speech_rate)
    pause_count_norm = max(0.0, min(1.0, 1.0 - (pause_count / 15)))
    avg_pause_norm = max(0.0, min(1.0, 1.0 - (avg_pause / 2.0)))
    pause_norm = (pause_count_norm + avg_pause_norm) / 2
    audio_quality_norm = max(0.0, min(1.0, (audio_quality - 4.0) / 4.0))

    composite = (
        articulation_norm * 0.25
        + stress_norm * 0.20
        + intonation_norm * 0.15
        + speech_rate_norm * 0.15
        + pause_norm * 0.15
        + audio_quality_norm * 0.10
    )

    score = _composite_to_pronunciation_band(composite)

    # Low ASR confidence means word-level analysis (phoneme/stress accuracy,
    # derived from forced-alignment against the transcript) can't be
    # trusted - pull the estimate toward a neutral midpoint rather than
    # trusting a potentially noisy extreme reading. This uses ASR confidence
    # only as a reliability gate on the acoustic measurements themselves,
    # not the transcript's wording/content.
    if asr_confidence < 0.7:
        score = (score * 0.6) + (5.5 * 0.4)

    return round(max(1.0, min(9.0, score)), 1)
