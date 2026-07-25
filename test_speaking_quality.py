#!/usr/bin/env python3
"""
Test script to validate speaking evaluator improvements
"""
import requests
import json
import time

def test_speaking_evaluation():
    """Test with realistic speaking data"""
    
    test_data = {
        "module": "speaking",
        "part_1": {
            "transcript": "I am from Tokyo. Yes, I enjoy living here very much. My family is here and I have good friends. I work in an office. I sit at a desk most of the time. Sometimes I work on my computer all day long. Sometimes it becomes repetitive so I go for a short walk to rest for a while.",
            "audio_metrics": {},
            "time_seconds": 90
        },
        "part_2": {
            "transcript": "I would like to talk about developing a habit of exercising regularly. This is an important topic for me because I desk job where I sit for many hours every day. I lack motivation to start. The challenge is managing time after work. But exercise has many benefits. It can help me gain energy level and reduce stress. It can also help me manage my weight which is important for my health.",
            "audio_metrics": {},
            "time_seconds": 120
        },
        "part_3": {
            "transcript": "People should maintain healthy lifestyles. Technology has made many jobs sedentary. But there are many unhealthy lifestyles nowadays. I think government should provide public gyms and parks to encourage activity. Awareness programs are important. We should promote physical activity through schools. The problem about urban lifestyle is people become busy so they eat fast food. This creates unhealthy habits. More infrastructure is needed to help people exercise properly.",
            "audio_metrics": {},
            "time_seconds": 150
        }
    }
    
    url = "http://127.0.0.1:60887/speaking/evaluate"
    
    print("=" * 80)
    print("SPEAKING EVALUATOR TEST")
    print("=" * 80)
    print("\n📤 Sending request with realistic speaking data...")
    print(f"Part 1: {len(test_data['part_1']['transcript'])} chars")
    print(f"Part 2: {len(test_data['part_2']['transcript'])} chars")
    print(f"Part 3: {len(test_data['part_3']['transcript'])} chars")
    
    try:
        response = requests.post(url, json=test_data, timeout=60)
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Print full result
            print("\n" + "=" * 80)
            print("📋 FULL EVALUATION RESULT")
            print("=" * 80)
            print(json.dumps(result, indent=2))
            
            # Validation checks
            print("\n" + "=" * 80)
            print("✅ VALIDATION CHECKS")
            print("=" * 80)
            
            checks = {
                "Overall band is reasonable": result.get("overall_band", 0) >= 4.0,
                "CEFR level is set": bool(result.get("cefr_level")),
                "Part 1 has non-zero scores": any([
                    result.get("part_1", {}).get("fluency", 0) > 0,
                    result.get("part_1", {}).get("lexical", 0) > 0,
                    result.get("part_1", {}).get("grammar", 0) > 0,
                ]),
                "Part 1 feedback strengths not empty": bool(result.get("part_1", {}).get("feedback", {}).get("strengths", "").strip()),
                "Part 1 good_usage populated": len(result.get("part_1", {}).get("vocabulary_feedback", {}).get("good_usage", [])) > 0,
                "Part 2 has non-zero scores": any([
                    result.get("part_2", {}).get("fluency", 0) > 0,
                    result.get("part_2", {}).get("lexical", 0) > 0,
                ]),
                "Part 2 feedback improvements not empty": bool(result.get("part_2", {}).get("feedback", {}).get("improvements", "").strip()),
                "Part 3 has non-zero scores": any([
                    result.get("part_3", {}).get("fluency", 0) > 0,
                    result.get("part_3", {}).get("lexical", 0) > 0,
                ]),
                "vocabulary_to_learn has items": len(result.get("vocabulary_to_learn", [])) >= 10,
            }
            
            pass_count = sum(1 for v in checks.values() if v)
            total_count = len(checks)
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check}")
            
            print(f"\n📊 Score: {pass_count}/{total_count} checks passed")
            
            # Quality assessment
            print("\n" + "=" * 80)
            print("📝 QUALITY ASSESSMENT")
            print("=" * 80)
            
            p1_strengths = result.get("part_1", {}).get("feedback", {}).get("strengths", "")
            p1_improvements = result.get("part_1", {}).get("feedback", {}).get("improvements", "")
            
            if p1_strengths and len(p1_strengths) > 50:
                print("✅ Part 1 feedback is detailed and specific")
            elif p1_strengths:
                print("⚠️  Part 1 feedback is brief, could be more detailed")
            else:
                print("❌ Part 1 feedback is empty")
            
            boilerplate = "Reduce hesitation and improve continuity of speech to enhance fluency."
            if p1_improvements and p1_improvements != boilerplate and len(p1_improvements) > 20:
                print("✅ Part 1 improvements feedback is specific and actionable")
            else:
                print("❌ Part 1 improvements feedback is generic or boilerplate")
            
            p1_good_usage = result.get("part_1", {}).get("vocabulary_feedback", {}).get("good_usage", [])
            actual_phrases = [u for u in p1_good_usage if " " in str(u) and len(u) > 3]
            if actual_phrases:
                print(f"✅ Part 1 has {len(actual_phrases)} actual phrases from transcript")
            else:
                print("❌ Part 1 good_usage contains random words, not phrases")
                
        else:
            print(f"\n❌ Error: Status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        print("Make sure the server is running on port 60887")

if __name__ == "__main__":
    print("Waiting for server to be ready...")
    time.sleep(2)
    test_speaking_evaluation()
