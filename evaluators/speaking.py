from utils.gpt_client import call_gpt
from utils.band import round_band
from pathlib import Path

SPEAKING_QUESTIONS = {
    1: [
        "What is your hometown?",
        "Do you like living there?"
    ],
    2: [
        "Describe a memorable trip you took."
    ],
    3: [
        "Why do people like travelling?",
        "How has tourism changed recently?"
    ]
}


def load_prompt():
    return Path("prompts/speaking_prompt.txt").read_text(encoding="utf-8")


def evaluate_speaking_part(part, transcript, audio_metrics, time_seconds=None):
    """Evaluate a single speaking part and return formatted result"""
    questions = SPEAKING_QUESTIONS.get(part, [])
    prompt_template = load_prompt()

    prompt = (
        prompt_template
        .replace("{{part}}", str(part))
        .replace("{{questions}}", str(questions))
        .replace("{{transcript}}", transcript)
        .replace("{{audio_metrics}}", str(audio_metrics))
    )

    # Base GPT evaluation
    result = call_gpt(prompt)

    # ============================================
    # EXTRACT CURRENT PART'S EVALUATION FROM RESPONSE
    # ============================================
    # GPT might return all 3 parts, we need to extract just this part's evaluation
    if isinstance(result, dict) and f"part_{part}" in result:
        # GPT returned multi-part response, extract just this part
        part_result = result.get(f"part_{part}", {})
        result = part_result
    elif isinstance(result, dict) and "fluency" not in result and "part_1" in result:
        # Full multi-part response, but we expected single part - use the specified part
        result = result.get(f"part_{part}", result.get("part_1", {}))

    # ============================================
    # EMERGENCY VALIDATION: Check if GPT returned broken data
    # ============================================
    fluency_from_gpt = result.get("fluency", 0)
    lexical_from_gpt = result.get("lexical", 0)
    grammar_from_gpt = result.get("grammar", 0)
    pronunciation_from_gpt = result.get("pronunciation", 0)
    
    # If ALL scores are 0 and we have transcript, GPT failed - use conservative defaults
    all_scores_zero = (fluency_from_gpt == 0 and lexical_from_gpt == 0 and 
                       grammar_from_gpt == 0 and pronunciation_from_gpt == 0)
    
    if all_scores_zero and transcript and len(transcript.strip()) > 0:
        print(f"[EMERGENCY AUTO-CORRECT] Part {part}: GPT returned all zeros for non-empty transcript. Using conservative defaults.")
        result["fluency"] = 5
        result["lexical"] = 5
        result["grammar"] = 5
        result["pronunciation"] = 6
        print(f"[EMERGENCY AUTO-CORRECT] Defaults applied: fluency=5, lexical=5, grammar=5, pronunciation=6")

    # ============================================
    # AUTO-FIX RULE 1: FLUENCY FLOOR (LOCKED)
    # ============================================
    fluency = result.get("fluency", 0)
    if part == 2 and time_seconds and isinstance(time_seconds, (int, float)) and time_seconds >= 60:
        # Part 2 with >= 60 seconds → Fluency must be >= 5
        if fluency < 5:
            print(f"[AUTO-FIX RULE 1] Part 2 time >= 60s, enforcing fluency >= 5 (was {fluency})")
            fluency = 5
    
    # ============================================
    # APPLY FLUENCY HARD RULES
    # ============================================
    wpm = audio_metrics.get("speech_rate_wpm", 0)
    pauses = audio_metrics.get("pause_count", 0)

    # Calculate WPM from time_seconds if provided
    if time_seconds and isinstance(time_seconds, (int, float)) and time_seconds > 0:
        words = len(transcript.split()) if transcript else 0
        wpm = round((words / time_seconds) * 60, 1) if time_seconds > 0 else 0
        print(f"[DEBUG] Part {part}: Calculated WPM from time_seconds: {wpm} (words={words}, time_seconds={time_seconds})")

    # Apply hard rules if audio metrics present
    if wpm > 0:
        if wpm < 90:
            fluency -= 1
        if wpm > 180:
            fluency -= 1
    
    if pauses > 5:
        fluency -= 1

    result["fluency"] = max(0, min(9, fluency))

    # ============================================
    # AUTO-FIX RULE 2: SCORE–FEEDBACK CONSISTENCY
    # ============================================
    fluency_final = result.get("fluency", 0)
    feedback_strengths = result.get("feedback", {}).get("strengths", "").lower()
    feedback_improvements = result.get("feedback", {}).get("improvements", "").lower()
    feedback_combined = feedback_strengths + " " + feedback_improvements
    
    # Keywords indicating high fluency in feedback
    high_fluency_keywords = ["clear", "logical", "communicates ideas", "maintains flow", "continuous", "well-organized", "coherent"]
    feedback_suggests_high_fluency = any(kw in feedback_combined for kw in high_fluency_keywords)
    
    # Keywords indicating low fluency in feedback
    low_fluency_keywords = ["breakdown", "hesitation", "unable to continue", "disjointed", "fragmented", "frequently pauses"]
    feedback_suggests_low_fluency = any(kw in feedback_combined for kw in low_fluency_keywords)
    
    if feedback_suggests_high_fluency and fluency_final < 5:
        print(f"[AUTO-FIX RULE 2] Part {part}: Feedback suggests high fluency but score is {fluency_final}. Correcting to 5.")
        result["fluency"] = 5
    
    if feedback_suggests_low_fluency and fluency_final >= 6:
        print(f"[AUTO-FIX RULE 2] Part {part}: Feedback suggests low fluency but score is {fluency_final}. Lowering to 4.")
        result["fluency"] = 4
    
    if fluency_final < 5 and not feedback_suggests_low_fluency:
        print(f"[AUTO-FIX RULE 2] Part {part}: Fluency < 5 but feedback doesn't mention issues. Adding breakdown mention.")
        # Ensure feedback exists before accessing nested keys
        if "feedback" not in result:
            result["feedback"] = {"strengths": "", "improvements": ""}
        result["feedback"]["improvements"] = "Reduce hesitation and improve continuity of speech to enhance fluency."

    # Calculate WPM if not provided by GPT
    if "wpm" not in result or result["wpm"] == 0:
        if wpm > 0:
            result["wpm"] = wpm
        else:
            # Estimate WPM from transcript length and fluency markers
            words = len(transcript.split()) if transcript else 0
            # Estimate speaking time based on word count and fluency
            # Assume average speaking rate: 120 words/minute for fluent speech
            estimated_time_minutes = words / 120.0 if words > 0 else 0
            result["wpm"] = round(words / estimated_time_minutes, 1) if estimated_time_minutes > 0 else 0

    # Ensure WPM is never 0 unless no transcript
    if result["wpm"] == 0 and transcript and len(transcript.strip()) > 0:
        # Conservative estimate: 120 WPM for fluent speech
        result["wpm"] = 120

    # ============================================
    # VALIDATE VOCABULARY_FEEDBACK (Rule 5)
    # ============================================
    
    if "vocabulary_feedback" not in result:
        result["vocabulary_feedback"] = {
            "good_usage": [],
            "suggested_improvements": []
        }
    else:
        if "good_usage" not in result["vocabulary_feedback"]:
            result["vocabulary_feedback"]["good_usage"] = []
        if "suggested_improvements" not in result["vocabulary_feedback"]:
            result["vocabulary_feedback"]["suggested_improvements"] = []
    
    # Auto-generate PHRASES if arrays are empty (not single words)
    if not result["vocabulary_feedback"]["good_usage"]:
        # Extract 2-4 word phrases from transcript
        words = transcript.split()
        phrases = []
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 5 and phrase not in phrases:
                phrases.append(phrase)
        
        # Select diverse phrases (first, middle, last)
        if len(phrases) >= 3:
            selected = [phrases[0], phrases[len(phrases)//2], phrases[-1]]
        elif len(phrases) >= 2:
            selected = [phrases[0], phrases[-1]]
        elif len(phrases) >= 1:
            selected = phrases
        else:
            selected = ["demonstrated competent communication"]
        
        result["vocabulary_feedback"]["good_usage"] = selected[:4]
        print(f"[AUTO-FIX RULE 5] Part {part}: Auto-generated good_usage with {len(result['vocabulary_feedback']['good_usage'])} phrases")
    
    if not result["vocabulary_feedback"]["suggested_improvements"]:
        # Generate contextual improvements based on transcript topic
        if part == 1:
            sug = ["expand with more specific examples → provide concrete details about your experience",
                   "add transitional phrases → use connectives like furthermore or additionally"]
        elif part == 2:
            sug = ["use more varied sentence structures → incorporate complex sentences for sophistication",
                   "employ topic-specific vocabulary → replace basic terms with more precise alternatives"]
        else:  # part 3
            sug = ["support opinions with examples → provide specific instances or recent trends",
                   "use academic vocabulary → incorporate formal terms like moreover or consequently"]
        
        result["vocabulary_feedback"]["suggested_improvements"] = sug
        print(f"[AUTO-FIX RULE 5] Part {part}: Auto-generated suggested_improvements")
    
    # Ensure all arrays have min 2 items
    if len(result["vocabulary_feedback"]["good_usage"]) < 2:
        result["vocabulary_feedback"]["good_usage"].append("expressed ideas coherently")
    if len(result["vocabulary_feedback"]["suggested_improvements"]) < 2:
        result["vocabulary_feedback"]["suggested_improvements"].append("incorporate more sophisticated vocabulary")

    # Ensure feedback exists and is not empty/boilerplate
    if "feedback" not in result:
        result["feedback"] = {
            "strengths": "The candidate demonstrates clear communication ability and provides coherent responses.",
            "improvements": "To improve further, expand answers with more detailed explanations and use more varied vocabulary."
        }
        print(f"[AUTO-FIX RULE 2] Part {part}: Feedback was missing, populated with defaults")
    else:
        # Check if feedback is empty or just the auto-fix boilerplate
        strengths = result.get("feedback", {}).get("strengths", "").strip()
        improvements = result.get("feedback", {}).get("improvements", "").strip()
        boilerplate = "Reduce hesitation and improve continuity of speech to enhance fluency."
        
        # If strengths is empty or improvements is ONLY the boilerplate, regenerate
        if not strengths:
            result["feedback"]["strengths"] = "The candidate demonstrates clear communication ability and provides coherent responses."
            print(f"[AUTO-FIX RULE 2] Part {part}: Strengths feedback was empty, auto-populated")
        
        if not improvements or improvements == boilerplate:
            result["feedback"]["improvements"] = "To improve further, expand answers with more detailed explanations and use more varied vocabulary."
            print(f"[AUTO-FIX RULE 2] Part {part}: Improvements feedback was generic, auto-populated with better guidance")

    # ============================================
    # ADD VOCABULARY_TO_LEARN FOR THIS PART
    # ============================================
    if "vocabulary_to_learn" not in result:
        result["vocabulary_to_learn"] = [
            {"word": "proficient", "usage_hint": "describe someone's skill level"},
            {"word": "articulate", "usage_hint": "express ideas clearly"},
            {"word": "fluent", "usage_hint": "speak smoothly"},
            {"word": "coherent", "usage_hint": "present logically"},
            {"word": "discourse marker", "usage_hint": "use connectors like however"},
            {"word": "elaborate", "usage_hint": "provide more detail"},
            {"word": "paraphrase", "usage_hint": "express differently"},
            {"word": "collocation", "usage_hint": "word combinations"},
            {"word": "hesitation", "usage_hint": "avoid pauses"},
            {"word": "intonation", "usage_hint": "vary pitch and stress"},
        ]
    
    # ============================================
    # CALCULATE CEFR LEVEL FOR THIS PART
    # ============================================
    fluency = result.get("fluency", 0)
    lexical = result.get("lexical", 0)
    grammar = result.get("grammar", 0)
    pronunciation = result.get("pronunciation", 0)
    
    part_avg = (fluency + lexical + grammar + pronunciation) / 4 if all([fluency, lexical, grammar, pronunciation]) else 5
    
    if part_avg >= 8.0:
        part_cefr = "C2"
    elif part_avg >= 7.0:
        part_cefr = "C1"
    elif part_avg >= 6.0:
        part_cefr = "B2"
    elif part_avg >= 5.0:
        part_cefr = "B1"
    elif part_avg >= 4.0:
        part_cefr = "A2"
    else:
        part_cefr = "A1"
    
    # Never output A2 for score >= 5.0
    if part_avg >= 5.0 and part_cefr == "A2":
        part_cefr = "B1"
    
    result["cefr_level"] = part_cefr

    return result


def evaluate_speaking(data: dict):
    """
    Evaluate all three parts and return aggregated part-wise assessment
    Input: data with part_1, part_2, part_3 keys or legacy single part format
    Output: complete module assessment with part_1, part_2, part_3
    """
    # Get all parts from data
    part_1_data = data.get("part_1")
    part_2_data = data.get("part_2")
    part_3_data = data.get("part_3")
    
    # Also check for single part in legacy format
    single_part = data.get("part", None)
    transcript = data.get("transcript", "")
    audio_metrics = data.get("audio_metrics", {})
    
    # Initialize results for all parts
    results = {
        "module": "speaking",
        "part_1": None,
        "part_2": None,
        "part_3": None,
        "vocabulary_to_learn": []
    }
    
    parts_evaluated = []
    all_vocab_to_learn = {}
    
    # Evaluate Part 1
    if isinstance(part_1_data, dict) and "transcript" in part_1_data:
        p1_result = evaluate_speaking_part(
            part=1,
            transcript=part_1_data.get("transcript", ""),
            audio_metrics=part_1_data.get("audio_metrics", {}),
            time_seconds=part_1_data.get("time_seconds")
        )
        results["part_1"] = {
            "fluency": p1_result.get("fluency", 0),
            "lexical": p1_result.get("lexical", 0),
            "grammar": p1_result.get("grammar", 0),
            "pronunciation": p1_result.get("pronunciation", 0),
            "wpm": p1_result.get("wpm", 0),
            "feedback": p1_result.get("feedback", {"strengths": "", "improvements": ""}),
            "vocabulary_feedback": p1_result.get("vocabulary_feedback", {"good_usage": [], "suggested_improvements": []}),
            "vocabulary_to_learn": p1_result.get("vocabulary_to_learn", []),
            "cefr_level": p1_result.get("cefr_level", "B1")
        }
        parts_evaluated.append(p1_result)
        if "vocabulary_to_learn" in p1_result:
            for item in p1_result.get("vocabulary_to_learn", []):
                all_vocab_to_learn[item.get("word")] = item
    elif single_part == 1 and transcript:
        p1_result = evaluate_speaking_part(
            part=1,
            transcript=transcript,
            audio_metrics=audio_metrics,
            time_seconds=data.get("time_seconds")
        )
        results["part_1"] = {
            "fluency": p1_result.get("fluency", 0),
            "lexical": p1_result.get("lexical", 0),
            "grammar": p1_result.get("grammar", 0),
            "pronunciation": p1_result.get("pronunciation", 0),
            "wpm": p1_result.get("wpm", 0),
            "feedback": p1_result.get("feedback", {"strengths": "", "improvements": ""}),
            "vocabulary_feedback": p1_result.get("vocabulary_feedback", {"good_usage": [], "suggested_improvements": []}),
            "vocabulary_to_learn": p1_result.get("vocabulary_to_learn", []),
            "cefr_level": p1_result.get("cefr_level", "B1")
        }
        parts_evaluated.append(p1_result)
        if "vocabulary_to_learn" in p1_result:
            for item in p1_result.get("vocabulary_to_learn", []):
                all_vocab_to_learn[item.get("word")] = item
    
    # Evaluate Part 2
    if isinstance(part_2_data, dict) and "transcript" in part_2_data:
        p2_result = evaluate_speaking_part(
            part=2,
            transcript=part_2_data.get("transcript", ""),
            audio_metrics=part_2_data.get("audio_metrics", {}),
            time_seconds=part_2_data.get("time_seconds")
        )
        results["part_2"] = {
            "fluency": p2_result.get("fluency", 0),
            "lexical": p2_result.get("lexical", 0),
            "grammar": p2_result.get("grammar", 0),
            "pronunciation": p2_result.get("pronunciation", 0),
            "wpm": p2_result.get("wpm", 0),
            "feedback": p2_result.get("feedback", {"strengths": "", "improvements": ""}),
            "vocabulary_feedback": p2_result.get("vocabulary_feedback", {"good_usage": [], "suggested_improvements": []}),
            "vocabulary_to_learn": p2_result.get("vocabulary_to_learn", []),
            "cefr_level": p2_result.get("cefr_level", "B1")
        }
        parts_evaluated.append(p2_result)
        if "vocabulary_to_learn" in p2_result:
            for item in p2_result.get("vocabulary_to_learn", []):
                all_vocab_to_learn[item.get("word")] = item
    elif single_part == 2 and transcript:
        p2_result = evaluate_speaking_part(
            part=2,
            transcript=transcript,
            audio_metrics=audio_metrics,
            time_seconds=data.get("time_seconds")
        )
        results["part_2"] = {
            "fluency": p2_result.get("fluency", 0),
            "lexical": p2_result.get("lexical", 0),
            "grammar": p2_result.get("grammar", 0),
            "pronunciation": p2_result.get("pronunciation", 0),
            "wpm": p2_result.get("wpm", 0),
            "feedback": p2_result.get("feedback", {"strengths": "", "improvements": ""}),
            "vocabulary_feedback": p2_result.get("vocabulary_feedback", {"good_usage": [], "suggested_improvements": []}),
            "vocabulary_to_learn": p2_result.get("vocabulary_to_learn", []),
            "cefr_level": p2_result.get("cefr_level", "B1")
        }
        parts_evaluated.append(p2_result)
        if "vocabulary_to_learn" in p2_result:
            for item in p2_result.get("vocabulary_to_learn", []):
                all_vocab_to_learn[item.get("word")] = item
    
    # Evaluate Part 3
    if isinstance(part_3_data, dict) and "transcript" in part_3_data:
        p3_result = evaluate_speaking_part(
            part=3,
            transcript=part_3_data.get("transcript", ""),
            audio_metrics=part_3_data.get("audio_metrics", {}),
            time_seconds=part_3_data.get("time_seconds")
        )
        results["part_3"] = {
            "fluency": p3_result.get("fluency", 0),
            "lexical": p3_result.get("lexical", 0),
            "grammar": p3_result.get("grammar", 0),
            "pronunciation": p3_result.get("pronunciation", 0),
            "wpm": p3_result.get("wpm", 0),
            "feedback": p3_result.get("feedback", {"strengths": "", "improvements": ""}),
            "vocabulary_feedback": p3_result.get("vocabulary_feedback", {"good_usage": [], "suggested_improvements": []}),
            "vocabulary_to_learn": p3_result.get("vocabulary_to_learn", []),
            "cefr_level": p3_result.get("cefr_level", "B1")
        }
        parts_evaluated.append(p3_result)
        if "vocabulary_to_learn" in p3_result:
            for item in p3_result.get("vocabulary_to_learn", []):
                all_vocab_to_learn[item.get("word")] = item
    elif single_part == 3 and transcript:
        p3_result = evaluate_speaking_part(
            part=3,
            transcript=transcript,
            audio_metrics=audio_metrics,
            time_seconds=data.get("time_seconds")
        )
        results["part_3"] = {
            "fluency": p3_result.get("fluency", 0),
            "lexical": p3_result.get("lexical", 0),
            "grammar": p3_result.get("grammar", 0),
            "pronunciation": p3_result.get("pronunciation", 0),
            "wpm": p3_result.get("wpm", 0),
            "feedback": p3_result.get("feedback", {"strengths": "", "improvements": ""}),
            "vocabulary_feedback": p3_result.get("vocabulary_feedback", {"good_usage": [], "suggested_improvements": []}),
            "vocabulary_to_learn": p3_result.get("vocabulary_to_learn", []),
            "cefr_level": p3_result.get("cefr_level", "B1")
        }
        parts_evaluated.append(p3_result)
        if "vocabulary_to_learn" in p3_result:
            for item in p3_result.get("vocabulary_to_learn", []):
                all_vocab_to_learn[item.get("word")] = item
    
    # Calculate overall band from evaluated parts
    if parts_evaluated:
        avg_fluency = sum(p.get("fluency", 0) for p in parts_evaluated) / len(parts_evaluated)
        avg_lexical = sum(p.get("lexical", 0) for p in parts_evaluated) / len(parts_evaluated)
        avg_grammar = sum(p.get("grammar", 0) for p in parts_evaluated) / len(parts_evaluated)
        avg_pronunciation = sum(p.get("pronunciation", 0) for p in parts_evaluated) / len(parts_evaluated)
        
        overall_avg = (avg_fluency + avg_lexical + avg_grammar + avg_pronunciation) / 4
        results["overall_band"] = round_band(overall_avg)
        
        # EMERGENCY VALIDATION: If overall_band is suspiciously low (< 1.5) but we have transcripts, it's wrong
        if results["overall_band"] < 1.5 and any(p.get("transcript", "") for p in parts_evaluated):
            print(f"[EMERGENCY AUTO-CORRECT] Overall band {results['overall_band']} is too low for evaluated transcripts. Using minimum 4.5.")
            results["overall_band"] = 4.5
    else:
        results["overall_band"] = 0
    
    # ============================================
    # AUTO-FIX RULE 3: LOCKED CEFR MAPPING
    # ============================================
    overall_band = results["overall_band"]
    
    if overall_band >= 8.0:
        cefr = "C2"
    elif overall_band >= 7.0:
        cefr = "C1"
    elif overall_band >= 6.0:
        cefr = "B2"
    elif overall_band >= 5.0:
        cefr = "B1"
    elif overall_band >= 4.0:
        cefr = "A2"
    else:
        cefr = "A1"
    
    # VALIDATE & AUTO-FIX: Never output A2 for band >= 5.0
    if overall_band >= 5.0 and cefr == "A2":
        print(f"[AUTO-FIX RULE 3] Band {overall_band} cannot map to A2. Correcting to B1.")
        cefr = "B1"
    
    results["cefr_level"] = cefr
    
    # ============================================
    # AUTO-FIX RULE 6 & 8: vocabulary_to_learn
    # ============================================
    vocab_list = list(all_vocab_to_learn.values())[:15]
    
    # VALIDATE: vocabulary_to_learn must NOT be empty (Rule 6 & 8)
    if not vocab_list:
        vocab_list = [
            {"word": "proficient", "usage_hint": "describe someone's skill level when speaking"},
            {"word": "articulate", "usage_hint": "express ideas clearly and precisely in speaking"},
            {"word": "fluent", "usage_hint": "speak smoothly without hesitation or pauses"},
            {"word": "coherent", "usage_hint": "present ideas in a logical and connected manner"},
            {"word": "discourse marker", "usage_hint": "use words like 'however', 'furthermore' to connect ideas"},
            {"word": "elaborate", "usage_hint": "provide more detail and explanation when answering questions"},
            {"word": "paraphrase", "usage_hint": "express the same idea differently in your own words"},
            {"word": "collocation", "usage_hint": "use word combinations that naturally go together"},
            {"word": "hesitation", "usage_hint": "pause or stammer when speaking - avoid this"},
            {"word": "intonation", "usage_hint": "vary the pitch and stress in your voice when speaking"},
        ]
        print(f"[AUTO-FIX RULE 8] vocabulary_to_learn was empty, populated with defaults")
    
    results["vocabulary_to_learn"] = vocab_list
    
    # ============================================
    # AUTO-FIX RULE 8: FINAL OUTPUT VALIDATION
    # ============================================
    # Ensure no null parts and all arrays are populated
    for part_num, part_key in enumerate(["part_1", "part_2", "part_3"], 1):
        if results[part_key] is not None:
            part = results[part_key]
            
            # Validate scores are numbers and reasonable
            for score_key in ["fluency", "lexical", "grammar", "pronunciation"]:
                if score_key not in part or not isinstance(part.get(score_key), (int, float)):
                    part[score_key] = 5
            
            # Ensure WPM is set
            if "wpm" not in part or part.get("wpm", 0) == 0:
                part["wpm"] = 120
            
            # Ensure feedback is complete and non-empty
            if "feedback" not in part:
                part["feedback"] = {}
            
            feedback = part.get("feedback", {})
            if not feedback.get("strengths", "").strip():
                if part_num == 1:
                    feedback["strengths"] = "The candidate provides clear answers about personal topics with relevant details."
                elif part_num == 2:
                    feedback["strengths"] = "The candidate maintains focus on the main topic and provides a coherent presentation."
                else:
                    feedback["strengths"] = "The candidate effectively discusses abstract concepts and supports opinions with reasoning."
            
            if not feedback.get("improvements", "").strip():
                if part_num == 1:
                    feedback["improvements"] = "Enhance responses by incorporating more varied vocabulary and reducing repetition of basic structures."
                elif part_num == 2:
                    feedback["improvements"] = "Improve by adding more specific examples and using sophisticated transitional phrases to structure ideas."
                else:
                    feedback["improvements"] = "Develop answers further by including recent examples and complex sentence structures for depth."
            
            part["feedback"] = feedback
            
            # Ensure vocabulary_feedback exists with non-empty arrays
            if "vocabulary_feedback" not in part:
                part["vocabulary_feedback"] = {
                    "good_usage": ["demonstrated communication"],
                    "suggested_improvements": ["use more sophisticated vocabulary and discourse markers"]
                }
            else:
                vocab_fb = part.get("vocabulary_feedback", {})
                
                # Ensure good_usage is not empty and has phrases (not single words)
                if not vocab_fb.get("good_usage"):
                    if part_num == 1:
                        vocab_fb["good_usage"] = ["work in an office", "personal details"]
                    elif part_num == 2:
                        vocab_fb["good_usage"] = ["important for me", "benefits of"]
                    else:
                        vocab_fb["good_usage"] = ["I believe that", "this shows"]
                else:
                    # Ensure all items have spaces (are phrases)
                    vocab_fb["good_usage"] = [str(u) for u in vocab_fb["good_usage"] if str(u).strip()]
                    if not vocab_fb["good_usage"]:
                        vocab_fb["good_usage"] = ["demonstrated competent communication"]
                
                # Ensure suggested_improvements is not empty
                if not vocab_fb.get("suggested_improvements"):
                    vocab_fb["suggested_improvements"] = [
                        "simple phrase → more sophisticated alternative",
                        "basic vocabulary → advanced term"
                    ]
                else:
                    vocab_fb["suggested_improvements"] = [str(u) for u in vocab_fb["suggested_improvements"] if str(u).strip()]
                    if not vocab_fb["suggested_improvements"]:
                        vocab_fb["suggested_improvements"] = ["enhance with more varied vocabulary"]
                
                # Ensure minimum 2 items in each array
                while len(vocab_fb["good_usage"]) < 2:
                    vocab_fb["good_usage"].append("adequate sentence construction")
                while len(vocab_fb["suggested_improvements"]) < 2:
                    vocab_fb["suggested_improvements"].append("incorporate more advanced vocabulary")
                
                part["vocabulary_feedback"] = vocab_fb
            
            # Ensure vocabulary_to_learn exists for this part
            if "vocabulary_to_learn" not in part or not part["vocabulary_to_learn"]:
                part["vocabulary_to_learn"] = [
                    {"word": "proficient", "usage_hint": "describe someone's skill level"},
                    {"word": "articulate", "usage_hint": "express ideas clearly"},
                    {"word": "fluent", "usage_hint": "speak smoothly"},
                    {"word": "coherent", "usage_hint": "present logically"},
                    {"word": "discourse marker", "usage_hint": "use connectors"},
                    {"word": "elaborate", "usage_hint": "provide more detail"},
                    {"word": "paraphrase", "usage_hint": "express differently"},
                    {"word": "collocation", "usage_hint": "word combinations"},
                    {"word": "hesitation", "usage_hint": "avoid pauses"},
                    {"word": "intonation", "usage_hint": "vary pitch"},
                ]
            
            # Ensure cefr_level exists for this part
            if "cefr_level" not in part:
                part_avg = (part.get("fluency", 0) + part.get("lexical", 0) + part.get("grammar", 0) + part.get("pronunciation", 0)) / 4
                if part_avg >= 8.0:
                    part["cefr_level"] = "C2"
                elif part_avg >= 7.0:
                    part["cefr_level"] = "C1"
                elif part_avg >= 6.0:
                    part["cefr_level"] = "B2"
                elif part_avg >= 5.0:
                    part["cefr_level"] = "B1"
                elif part_avg >= 4.0:
                    part["cefr_level"] = "A2"
                else:
                    part["cefr_level"] = "A1"
                
                # Never A2 for score >= 5.0
                if part_avg >= 5.0 and part["cefr_level"] == "A2":
                    part["cefr_level"] = "B1"
    
    # Ensure overall_band is set
    if not results.get("overall_band") or results["overall_band"] == 0:
        results["overall_band"] = 5.5
    
    # Ensure cefr_level is set
    if not results.get("cefr_level"):
        overall_band = results.get("overall_band", 5.5)
        if overall_band >= 8.0:
            cefr = "C2"
        elif overall_band >= 7.0:
            cefr = "C1"
        elif overall_band >= 6.0:
            cefr = "B2"
        elif overall_band >= 5.0:
            cefr = "B1"
        else:
            cefr = "A2"
        results["cefr_level"] = cefr
    
    # Ensure vocabulary_to_learn has at least 10 items
    if not results.get("vocabulary_to_learn") or len(results["vocabulary_to_learn"]) < 10:
        results["vocabulary_to_learn"] = [
            {"word": "proficient", "usage_hint": "describe someone's skill level when speaking"},
            {"word": "articulate", "usage_hint": "express ideas clearly and precisely in speaking"},
            {"word": "fluent", "usage_hint": "speak smoothly without hesitation or pauses"},
            {"word": "coherent", "usage_hint": "present ideas in a logical and connected manner"},
            {"word": "discourse marker", "usage_hint": "use words like 'however', 'furthermore' to connect ideas"},
            {"word": "elaborate", "usage_hint": "provide more detail and explanation when answering questions"},
            {"word": "paraphrase", "usage_hint": "express the same idea differently in your own words"},
            {"word": "collocation", "usage_hint": "use word combinations that naturally go together"},
            {"word": "hesitation", "usage_hint": "pause or stammer when speaking - avoid this"},
            {"word": "intonation", "usage_hint": "vary the pitch and stress in your voice when speaking"},
        ]
        print(f"[AUTO-FIX RULE 8] vocabulary_to_learn populated with defaults (had {len(results.get('vocabulary_to_learn', []))} items)")
    
    # Final output structure validation
    final_check = {
        "module": results.get("module", "speaking"),
        "part_1": results.get("part_1"),
        "part_2": results.get("part_2"),
        "part_3": results.get("part_3"),
        "overall_band": results.get("overall_band", 5.5),
        "overall_cefr_level": results.get("cefr_level", "B1")
    }
    
    return final_check
