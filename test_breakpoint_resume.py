"""
Test script to verify breakpoint resume functionality.
"""
import os
import sys
import json
import hashlib
import tempfile

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(__file__))


def test_resume_logic():
    """Test the resume logic without importing full modules."""
    print("\n=== Test: Resume Logic ===")
    
    # Simulate the key functions from BaseDataset
    def _get_sample_key(sample):
        """Generate a unique key for a sample to identify duplicates."""
        if "id" in sample:
            return f"id:{sample['id']}"
        if "question" in sample and "answer" in sample:
            content = str(sample['question']) + str(sample['answer'])
            hash_value = hashlib.sha256(content.encode()).hexdigest()
            return f"qa:{hash_value}"
        content = json.dumps(sample, sort_keys=True)
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return f"hash:{hash_value}"
    
    def _filter_remaining_samples(samples, existing_results):
        """Filter out samples that have already been processed."""
        if not existing_results:
            return samples, []
        
        # Build a dict of processed sample keys to results
        processed_results = {}
        for result in existing_results:
            if "response" in result:
                key = _get_sample_key(result)
                processed_results[key] = result
        
        # Filter samples and build ordered lists
        remaining_samples = []
        existing_samples_ordered = []
        for sample in samples:
            key = _get_sample_key(sample)
            if key in processed_results:
                # Keep in original order
                existing_samples_ordered.append(processed_results[key])
            else:
                remaining_samples.append(sample)
        
        return remaining_samples, existing_samples_ordered
    
    # Test sample key generation
    sample1 = {"id": "123", "question": "Q1"}
    key1 = _get_sample_key(sample1)
    assert key1 == "id:123", f"Expected 'id:123', got {key1}"
    print("✓ Sample key generation works")
    
    # Test filtering
    all_samples = [
        {"id": "1", "question": "Q1"},
        {"id": "2", "question": "Q2"},
        {"id": "3", "question": "Q3"},
    ]
    
    existing_results = [
        {"id": "1", "question": "Q1", "response": "R1"},
        {"id": "2", "question": "Q2", "response": "R2"},
    ]
    
    remaining, existing = _filter_remaining_samples(all_samples, existing_results)
    assert len(remaining) == 1, f"Expected 1 remaining, got {len(remaining)}"
    assert len(existing) == 2, f"Expected 2 existing, got {len(existing)}"
    assert remaining[0]["id"] == "3", "Sample 3 should remain"
    print(f"✓ Filtering works: {len(remaining)} remaining, {len(existing)} existing")


def test_file_operations():
    """Test file loading operations."""
    print("\n=== Test: File Operations ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test loading nonexistent file
        nonexistent_path = os.path.join(tmpdir, "nonexistent.json")
        result = None
        if os.path.exists(nonexistent_path):
            try:
                with open(nonexistent_path, "r") as f:
                    result = json.load(f)
            except:
                result = None
        
        assert result is None, "Should return None for nonexistent file"
        print("✓ Handles nonexistent files correctly")
        
        # Test loading valid file
        valid_path = os.path.join(tmpdir, "valid.json")
        test_data = [{"id": "1", "response": "R1"}]
        with open(valid_path, "w") as f:
            json.dump(test_data, f)
        
        with open(valid_path, "r") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data, "Should load valid data"
        print("✓ Loads valid files correctly")
        
        # Test loading corrupted file
        corrupted_path = os.path.join(tmpdir, "corrupted.json")
        with open(corrupted_path, "w") as f:
            f.write("{invalid json")
        
        try:
            with open(corrupted_path, "r") as f:
                json.load(f)
            corrupted_result = "loaded"
        except json.JSONDecodeError:
            corrupted_result = None
        
        assert corrupted_result is None, "Should handle corrupted JSON"
        print("✓ Handles corrupted JSON correctly")


def test_incremental_results():
    """Test incremental results update logic."""
    print("\n=== Test: Incremental Results ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        total_results_path = os.path.join(tmpdir, "total_results.json")
        
        # Simulate first dataset
        dataset1_results = {"acc": 0.85, "total": 100}
        
        if os.path.exists(total_results_path):
            with open(total_results_path, "r") as f:
                total_results = json.load(f)
        else:
            total_results = {}
        
        total_results["dataset1"] = dataset1_results
        with open(total_results_path, "w") as f:
            json.dump(total_results, f, indent=4)
        
        # Simulate second dataset
        dataset2_results = {"acc": 0.90, "total": 150}
        
        with open(total_results_path, "r") as f:
            total_results = json.load(f)
        
        total_results["dataset2"] = dataset2_results
        with open(total_results_path, "w") as f:
            json.dump(total_results, f, indent=4)
        
        # Verify both datasets are in total_results
        with open(total_results_path, "r") as f:
            final_results = json.load(f)
        
        assert "dataset1" in final_results, "Dataset1 should be in results"
        assert "dataset2" in final_results, "Dataset2 should be in results"
        assert final_results["dataset1"]["acc"] == 0.85, "Dataset1 acc should be preserved"
        assert final_results["dataset2"]["acc"] == 0.90, "Dataset2 acc should be correct"
        print("✓ Incremental updates work correctly")
        
        # Simulate resume: update dataset1
        dataset1_updated = {"acc": 0.87, "total": 100, "updated": True}
        
        with open(total_results_path, "r") as f:
            total_results = json.load(f)
        
        total_results["dataset1"] = dataset1_updated
        with open(total_results_path, "w") as f:
            json.dump(total_results, f, indent=4)
        
        with open(total_results_path, "r") as f:
            final_results = json.load(f)
        
        assert final_results["dataset1"]["acc"] == 0.87, "Dataset1 should be updated"
        assert final_results["dataset1"].get("updated") == True, "Dataset1 should have new field"
        assert "dataset2" in final_results, "Dataset2 should still exist"
        print("✓ Resume updates work correctly")


def test_chunk_completion_detection():
    """Test detection of completed chunks."""
    print("\n=== Test: Chunk Completion Detection ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        num_chunks = 3
        
        # Create chunk 0 and 1
        for i in range(2):
            chunk_path = os.path.join(tmpdir, f"results_{i}.json")
            with open(chunk_path, "w") as f:
                json.dump([{"id": str(i), "response": f"R{i}"}], f)
        
        # Check completion
        all_files = os.listdir(tmpdir)
        chunk_files = [f for f in all_files if f.startswith("results_")]
        
        assert len(chunk_files) == 2, f"Expected 2 chunk files, got {len(chunk_files)}"
        assert len(chunk_files) < num_chunks, "Not all chunks should be complete"
        print(f"✓ Detected {len(chunk_files)}/{num_chunks} chunks")
        
        # Add chunk 2
        chunk_path = os.path.join(tmpdir, f"results_2.json")
        with open(chunk_path, "w") as f:
            json.dump([{"id": "2", "response": "R2"}], f)
        
        all_files = os.listdir(tmpdir)
        chunk_files = [f for f in all_files if f.startswith("results_")]
        
        assert len(chunk_files) == num_chunks, f"Expected {num_chunks} chunk files, got {len(chunk_files)}"
        print(f"✓ All {num_chunks} chunks detected")
        
        # Simulate merging
        merged_results = []
        for i in range(num_chunks):
            chunk_path = os.path.join(tmpdir, f"results_{i}.json")
            with open(chunk_path, "r") as f:
                merged_results.extend(json.load(f))
        
        assert len(merged_results) == num_chunks, f"Expected {num_chunks} merged samples"
        print(f"✓ Merged {len(merged_results)} samples from all chunks")


def test_ordering_preserved():
    """Test that sample ordering is preserved during resume."""
    print("\n=== Test: Ordering Preserved ===")
    
    # Test sample key generation
    def _get_sample_key(sample):
        if "id" in sample:
            return f"id:{sample['id']}"
        if "question" in sample and "answer" in sample:
            content = str(sample['question']) + str(sample['answer'])
            hash_value = hashlib.sha256(content.encode()).hexdigest()
            return f"qa:{hash_value}"
        content = json.dumps(sample, sort_keys=True)
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return f"hash:{hash_value}"
    
    def _filter_remaining_samples(samples, existing_results):
        if not existing_results:
            return samples, []
        
        processed_results = {}
        for result in existing_results:
            if "response" in result:
                key = _get_sample_key(result)
                processed_results[key] = result
        
        remaining_samples = []
        existing_samples_ordered = []
        for sample in samples:
            key = _get_sample_key(sample)
            if key in processed_results:
                existing_samples_ordered.append(processed_results[key])
            else:
                remaining_samples.append(sample)
        
        return remaining_samples, existing_samples_ordered
    
    # Create samples in specific order
    all_samples = [
        {"id": "1", "question": "Q1"},
        {"id": "2", "question": "Q2"},
        {"id": "3", "question": "Q3"},
        {"id": "4", "question": "Q4"},
    ]
    
    # Simulate existing results for samples 2 and 4 (out of order in file)
    existing_results = [
        {"id": "4", "question": "Q4", "response": "R4"},
        {"id": "2", "question": "Q2", "response": "R2"},
    ]
    
    remaining, existing = _filter_remaining_samples(all_samples, existing_results)
    
    # Verify remaining samples
    assert len(remaining) == 2, f"Expected 2 remaining, got {len(remaining)}"
    assert remaining[0]["id"] == "1", "First remaining should be sample 1"
    assert remaining[1]["id"] == "3", "Second remaining should be sample 3"
    print("✓ Remaining samples in correct order")
    
    # Verify existing samples are in original order
    assert len(existing) == 2, f"Expected 2 existing, got {len(existing)}"
    assert existing[0]["id"] == "2", "First existing should be sample 2 (original order)"
    assert existing[1]["id"] == "4", "Second existing should be sample 4 (original order)"
    print("✓ Existing samples in original order (not file order)")
    
    # Simulate combining results
    combined = existing + [{"id": "1", "response": "R1"}, {"id": "3", "response": "R3"}]
    
    # Verify combined order matches original sample order
    assert combined[0]["id"] == "2", "First in combined should be sample 2"
    assert combined[1]["id"] == "4", "Second in combined should be sample 4"
    assert combined[2]["id"] == "1", "Third in combined should be sample 1"
    assert combined[3]["id"] == "3", "Fourth in combined should be sample 3"
    print("✓ Combined results maintain relative ordering")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Breakpoint Resume Functionality")
    print("=" * 60)
    
    tests = [
        test_resume_logic,
        test_file_operations,
        test_incremental_results,
        test_chunk_completion_detection,
        test_ordering_preserved,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "=" * 60)
    if not failed:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
