#!/usr/bin/env python3
"""
Direct test of speaking evaluator without API
"""
import sys
import json
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file
project_root = Path(__file__).parent.resolve()
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)

print(f"[OK] API Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluators.speaking import evaluate_speaking

def test_speaking_direct():
    """Test speaking evaluator directly"""
    
    test_data = {
        "test_type": "speaking",
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
    
    print("=" * 80)
    print("DIRECT SPEAKING EVALUATOR TEST")
    print("=" * 80)
    print("\n🔍 Testing evaluate_speaking() directly...")
    
    try:
        result = evaluate_speaking(test_data)
        
        print("\n✅ Evaluation completed successfully!")
        print("\n" + "=" * 80)
        print("FULL RESULT")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        
        # Validation
        print("\n" + "=" * 80)
        print("✅ VALIDATION CHECKLIST")
        print("=" * 80)
        
        checks = {
            "✓ Module is 'speaking'": result.get("module") == "speaking",
            "✓ Overall band >= 4": result.get("overall_band", 0) >= 4.0,
            "✓ CEFR is set (B1-B2 range)": result.get("cefr_level") in ["B1", "B2"],
            "✓ Part 1 has scores": result.get("part_1", {}).get("fluency", 0) > 0,
            "✓ Part 1 strengths populated": bool(result.get("part_1", {}).get("feedback", {}).get("strengths", "").strip()),
            "✓ Part 1 improvements populated": bool(result.get("part_1", {}).get("feedback", {}).get("improvements", "").strip()),
            "✓ Part 1 good_usage has items": len(result.get("part_1", {}).get("vocabulary_feedback", {}).get("good_usage", [])) > 0,
            "✓ Part 2 has scores": result.get("part_2", {}).get("fluency", 0) > 0,
            "✓ Part 2 strengths populated": bool(result.get("part_2", {}).get("feedback", {}).get("strengths", "").strip()),
            "✓ Part 2 improvements populated": bool(result.get("part_2", {}).get("feedback", {}).get("improvements", "").strip()),
            "✓ Part 3 has scores": result.get("part_3", {}).get("fluency", 0) > 0,
            "✓ Part 3 strengths populated": bool(result.get("part_3", {}).get("feedback", {}).get("strengths", "").strip()),
            "✓ vocabulary_to_learn has 10+ items": len(result.get("vocabulary_to_learn", [])) >= 10,
        }
        
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        
        for check, result_val in checks.items():
            status = "✅" if result_val else "❌"
            print(f"{status} {check}")
        
        print(f"\n📊 Overall: {passed}/{total} checks passed")
        
        if passed == total:
            print("\n🎉 ALL CHECKS PASSED!")
        else:
            print(f"\n⚠️  {total - passed} checks failed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speaking_direct()
