import requests
import json
import time

# Wait for server to be ready
time.sleep(2)

# Test data
test_data = {
    "module": "speaking",
    "part_1": {
        "transcript": "I am from Tokyo. Yes, I enjoy living here because my family is close and I have good friends.",
        "audio_metrics": {},
        "time_seconds": 30
    },
    "part_2": {
        "transcript": "Let me tell you about a memorable trip to Kyoto last year. I visited many temples and traditional gardens. The weather was beautiful. It was an amazing experience.",
        "audio_metrics": {},
        "time_seconds": 90
    },
    "part_3": {
        "transcript": "Tourism has changed a lot in recent years. People now use the internet to book trips. More travelers visit destinations around the world. Some places become crowded.",
        "audio_metrics": {},
        "time_seconds": 50
    }
}

url = "http://127.0.0.1:60887/speaking/evaluate"

try:
    print("Sending request to speaking endpoint...")
    response = requests.post(url, json=test_data, timeout=45)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS! Response received:")
        print(json.dumps(result, indent=2))
        
        # Validate required fields
        print("\n📋 VALIDATION CHECKLIST:")
        print(f"✓ Module: {result.get('module')}")
        print(f"✓ Overall Band: {result.get('overall_band')}")
        print(f"✓ CEFR Level: {result.get('cefr_level')}")
        print(f"✓ Part 1: {'Yes' if result.get('part_1') else 'Missing'}")
        print(f"✓ Part 2: {'Yes' if result.get('part_2') else 'Missing'}")
        print(f"✓ Part 3: {'Yes' if result.get('part_3') else 'Missing'}")
        print(f"✓ Vocabulary to Learn: {len(result.get('vocabulary_to_learn', []))} items")
        
        # Check for empty arrays
        for part_num in [1, 2, 3]:
            part_key = f'part_{part_num}'
            if result.get(part_key):
                vocab_fb = result[part_key].get('vocabulary_feedback', {})
                print(f"✓ Part {part_num} good_usage: {len(vocab_fb.get('good_usage', []))} items")
                print(f"✓ Part {part_num} suggested_improvements: {len(vocab_fb.get('suggested_improvements', []))} items")
    else:
        print(f"❌ Error: Status code {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure the server is running on port 60887")
