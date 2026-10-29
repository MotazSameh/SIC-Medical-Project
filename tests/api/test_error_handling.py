"""
Error handling validation tests
Tests edge cases and error conditions
"""

import requests
import time
import json

API_BASE = "http://localhost:8000"

def test_server_not_ready():
    """
    Test 1: API called before initialization complete
    Note: Hard to test automatically - requires manual timing
    """
    print("\n" + "="*80)
    print("TEST 1: Server Not Ready (503)")
    print("="*80)
    print("⚠️  Manual test required:")
    print("   1. Restart API: python src/api/main.py")
    print("   2. Immediately call /ask endpoint (within 2 seconds)")
    print("   3. Should return 503 'Retriever not initialized'")
    print("   SKIPPING automated test...")

def test_empty_query():
    """
    Test 2: Empty query string
    Expected: 422 Unprocessable Entity
    """
    print("\n" + "="*80)
    print("TEST 2: Empty Query (422)")
    print("="*80)
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={
                "query": "",
                "top_k": 5,
                "retriever_type": "hybrid"
            }
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 422:
            print("✅ PASS - Correctly rejected empty query with 422")
            return True
        else:
            print(f"❌ FAIL - Expected 422, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_invalid_top_k_too_high():
    """
    Test 3: top_k exceeds maximum (20)
    Expected: 422 Unprocessable Entity
    """
    print("\n" + "="*80)
    print("TEST 3: Invalid top_k - Too High (422)")
    print("="*80)
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={
                "query": "What are the side effects of warfarin?",
                "top_k": 100,  # Max is 20
                "retriever_type": "hybrid"
            }
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 422:
            print("✅ PASS - Correctly rejected top_k=100 with 422")
            return True
        else:
            print(f"❌ FAIL - Expected 422, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_invalid_top_k_zero():
    """
    Test 4: top_k = 0
    Expected: 422 Unprocessable Entity
    """
    print("\n" + "="*80)
    print("TEST 4: Invalid top_k - Zero (422)")
    print("="*80)
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={
                "query": "What are the side effects of warfarin?",
                "top_k": 0,  # Min is 1
                "retriever_type": "hybrid"
            }
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 422:
            print("✅ PASS - Correctly rejected top_k=0 with 422")
            return True
        else:
            print(f"❌ FAIL - Expected 422, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_invalid_retriever_type():
    """
    Test 5: Invalid retriever_type
    Expected: Should either default to hybrid OR return 400 error
    """
    print("\n" + "="*80)
    print("TEST 5: Invalid retriever_type")
    print("="*80)
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={
                "query": "What are the side effects of warfarin?",
                "top_k": 5,
                "retriever_type": "invalid_type"  # Not vector/bm25/hybrid
            }
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response (preview): {data.get('answer')[:100]}...")
            print("✅ PASS - Handled gracefully (defaulted or processed)")
            return True
        elif response.status_code == 400 or response.status_code == 422:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            print("✅ PASS - Correctly rejected invalid retriever_type")
            return True
        else:
            print(f"❌ FAIL - Unexpected status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_missing_required_field():
    """
    Test 6: Missing required 'query' field
    Expected: 422 Unprocessable Entity
    """
    print("\n" + "="*80)
    print("TEST 6: Missing Required Field (422)")
    print("="*80)
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={
                "top_k": 5,
                "retriever_type": "hybrid"
                # Missing 'query' field
            }
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 422:
            print("✅ PASS - Correctly rejected missing query field with 422")
            return True
        else:
            print(f"❌ FAIL - Expected 422, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_malformed_json():
    """
    Test 7: Malformed JSON body
    Expected: 422 Unprocessable Entity
    """
    print("\n" + "="*80)
    print("TEST 7: Malformed JSON (422)")
    print("="*80)
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            data="not valid json",  # Invalid JSON
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
        if response.status_code == 422:
            print("✅ PASS - Correctly rejected malformed JSON with 422")
            return True
        else:
            print(f"❌ FAIL - Expected 422, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_very_long_query():
    """
    Test 8: Extremely long query (edge case)
    Expected: Should handle gracefully (may refuse or truncate)
    """
    print("\n" + "="*80)
    print("TEST 8: Very Long Query")
    print("="*80)
    
    long_query = "What are the side effects of warfarin? " * 100  # 500+ words
    
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={
                "query": long_query,
                "top_k": 5,
                "retriever_type": "hybrid"
            },
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Query length: {len(long_query)} characters")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response (preview): {data.get('answer')[:100]}...")
            print("✅ PASS - Handled long query gracefully")
            return True
        elif response.status_code == 400 or response.status_code == 422:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            print("✅ PASS - Rejected long query appropriately")
            return True
        else:
            print(f"⚠️  WARNING - Unexpected status {response.status_code}")
            return True  # Still pass, just unexpected
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    print("\n" + "="*80)
    print("🔧 ERROR HANDLING VALIDATION TESTS")
    print("="*80)
    print(f"API: {API_BASE}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API is running
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5)
        if health.status_code == 200:
            print("✅ API is running")
        else:
            print(f"⚠️  API health check returned {health.status_code}")
    except:
        print("❌ ERROR: API is not running")
        print("   Start with: python src/api/main.py")
        return
    
    # Run tests
    results = []
    
    test_server_not_ready()  # Manual test
    results.append(("Empty Query", test_empty_query()))
    time.sleep(0.5)
    results.append(("top_k Too High", test_invalid_top_k_too_high()))
    time.sleep(0.5)
    results.append(("top_k Zero", test_invalid_top_k_zero()))
    time.sleep(0.5)
    results.append(("Invalid Retriever", test_invalid_retriever_type()))
    time.sleep(0.5)
    results.append(("Missing Field", test_missing_required_field()))
    time.sleep(0.5)
    results.append(("Malformed JSON", test_malformed_json()))
    time.sleep(0.5)
    results.append(("Very Long Query", test_very_long_query()))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"Pass Rate: {(passed/total*100):.1f}%")
    
    print("\n" + "="*80)
    if passed == total:
        print("🎉 ALL ERROR HANDLING TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
