#!/usr/bin/env python
"""
Test Cerebras Cloud API Connection
===================================

Run this script to verify your Cerebras API key works and API calls are being made.

Usage:
    python test_cerebras_connection.py YOUR_API_KEY

Or set environment variable:
    set CEREBRAS_API_KEY=your_key_here
    python test_cerebras_connection.py
"""

import sys
import os
import time

def test_cerebras_connection(api_key: str = None):
    """Test if Cerebras API is properly connected."""
    
    print("=" * 60)
    print("CEREBRAS API CONNECTION TEST")
    print("=" * 60)
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get("CEREBRAS_API_KEY")
    
    if not api_key:
        print("\n❌ ERROR: No API key provided!")
        print("Usage: python test_cerebras_connection.py YOUR_API_KEY")
        print("Or set CEREBRAS_API_KEY environment variable")
        return False
    
    print(f"\n✅ API Key provided (length: {len(api_key)})")
    
    # Test 1: SDK Import
    print("\n[TEST 1] Checking Cerebras SDK installation...")
    try:
        from cerebras.cloud.sdk import Cerebras
        print("✅ Cerebras SDK is installed!")
    except ImportError:
        print("❌ Cerebras SDK NOT installed!")
        print("   Run: pip install cerebras-cloud-sdk")
        return False
    
    # Test 2: Create client
    print("\n[TEST 2] Creating Cerebras client...")
    try:
        client = Cerebras(api_key=api_key)
        print("✅ Client created successfully!")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False
    
    # Test 3: Make a simple API call
    print("\n[TEST 3] Making test API call to Cerebras Cloud...")
    print("   Model: llama3.3-70b")
    print("   Prompt: 'Say hello in one word'")
    
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model="llama3.3-70b",
            messages=[{"role": "user", "content": "Say hello in one word"}],
            max_tokens=10,
            temperature=0.0
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        response_text = response.choices[0].message.content
        print(f"✅ API CALL SUCCESSFUL!")
        print(f"   Response: {response_text}")
        print(f"   Latency: {latency_ms:.1f} ms")
        
    except Exception as e:
        print(f"❌ API call FAILED: {e}")
        return False
    
    # Test 4: Test CerebrasRiskEngine
    print("\n[TEST 4] Testing CerebrasRiskEngine for sepsis analysis...")
    try:
        from cerebras_inference import CerebrasRiskEngine
        
        engine = CerebrasRiskEngine(
            api_key=api_key,
            model_tier="fast",
            reasoning_cycles=3  # Small number for quick test
        )
        
        # Check if in simulation mode
        if engine._simulation_mode:
            print("⚠️ WARNING: CerebrasRiskEngine is in SIMULATION MODE!")
            print("   API calls will NOT be made")
        else:
            print("✅ CerebrasRiskEngine connected to Cerebras Cloud!")
            
            # Make a test analysis
            print("\n   Making test sepsis risk analysis...")
            test_vitals = {
                "heart_rate": 105,
                "map": 65,
                "respiratory_rate": 24,
                "spo2": 93,
                "temperature": 38.5,
                "lactate": 3.2,
                "wbc": 15
            }
            
            start_time = time.perf_counter()
            result = engine.analyze_patient(vitals=test_vitals)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            print(f"\n   ✅ ANALYSIS COMPLETE!")
            print(f"   Risk Score: {result.risk_score:.3f}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Key Factors: {result.key_factors}")
            print(f"   Model Used: {result.metrics.model_used}")
            print(f"   Reasoning Cycles: {result.metrics.reasoning_cycles}")
            print(f"   Latency: {latency_ms:.1f} ms")
            
            if result.metrics.model_used == "simulation":
                print("\n   ⚠️ WARNING: Result came from LOCAL SIMULATION, not Cerebras Cloud!")
            else:
                print("\n   ✅ Result came from CEREBRAS CLOUD API!")
            
    except Exception as e:
        print(f"❌ CerebrasRiskEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_cerebras_connection(api_key)
    sys.exit(0 if success else 1)
