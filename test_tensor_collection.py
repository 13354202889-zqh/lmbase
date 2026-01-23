import os
import shutil
import torch
from lmbase.utils.tools import BlockBasedStoreManager


def test_tensor_collection_storage():
    """Test the new functionality for storing tensor lists/tuples in folders."""
    folder = "EXPERIMENT/test_tensor_collection"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    
    manager = BlockBasedStoreManager(folder=folder, block_size=10)
    
    print("Testing tensor list storage...")
    
    # Create a list of tensors
    tensor_list = [torch.randn(3, 4), torch.randn(2, 5), torch.randn(1, 10)]
    
    # Save the tensor list
    manager.save("test_tensor_list", {"tensors": tensor_list})
    
    # Load the data back
    loaded_data = manager.load("test_tensor_list")
    
    print(f"Original tensor list length: {len(tensor_list)}")
    print(f"Loaded tensor list length: {len(loaded_data['tensors'])}")
    
    # Verify that the loaded data is a list of tensors
    assert isinstance(loaded_data['tensors'], list), "Loaded data should be a list"
    assert len(loaded_data['tensors']) == len(tensor_list), "Length should match"
    
    # Verify each tensor
    for i, (orig_tensor, loaded_tensor) in enumerate(zip(tensor_list, loaded_data['tensors'])):
        assert torch.equal(orig_tensor, loaded_tensor), f"Tensor {i} should match"
        print(f"Tensor {i} matches: shape {orig_tensor.shape}")
    
    print("Testing tensor tuple storage...")
    
    # Create a tuple of tensors
    tensor_tuple = (torch.ones(2, 2), torch.zeros(3, 3), torch.eye(2))
    
    # Save the tensor tuple
    manager.save("test_tensor_tuple", {"tensors": tensor_tuple})
    
    # Load the data back
    loaded_tuple_data = manager.load("test_tensor_tuple")
    
    print(f"Original tensor tuple length: {len(tensor_tuple)}")
    print(f"Loaded tensor tuple length: {len(loaded_tuple_data['tensors'])}")
    
    # Verify that the loaded data is a list of tensors (tuples become lists after JSON serialization)
    assert isinstance(loaded_tuple_data['tensors'], list), "Loaded data should be a list"
    assert len(loaded_tuple_data['tensors']) == len(tensor_tuple), "Length should match"
    
    # Verify each tensor
    for i, (orig_tensor, loaded_tensor) in enumerate(zip(tensor_tuple, loaded_tuple_data['tensors'])):
        assert torch.equal(orig_tensor, loaded_tensor), f"Tensor {i} should match"
        print(f"Tensor {i} matches: shape {orig_tensor.shape}")
    
    print("Testing mixed content (not all tensors)...")
    
    # Create a list with mixed content (not all tensors)
    mixed_list = [torch.randn(2, 2), "string", 42, torch.randn(3, 3)]
    
    # Save the mixed list
    manager.save("test_mixed_list", {"mixed": mixed_list})
    
    # Load the data back
    loaded_mixed_data = manager.load("test_mixed_list")
    
    print(f"Mixed list saved and loaded successfully")
    print(f"Type of first element: {type(loaded_mixed_data['mixed'][0])}")
    print(f"Type of second element: {type(loaded_mixed_data['mixed'][1])}")
    print(f"Type of third element: {type(loaded_mixed_data['mixed'][2])}")
    print(f"Type of fourth element: {type(loaded_mixed_data['mixed'][3])}")
    
    # Verify that the tensors are still properly loaded
    assert torch.equal(mixed_list[0], loaded_mixed_data['mixed'][0]), "First tensor should match"
    assert loaded_mixed_data['mixed'][1] == "string", "String should match"
    assert loaded_mixed_data['mixed'][2] == 42, "Number should match"
    assert torch.equal(mixed_list[3], loaded_mixed_data['mixed'][3]), "Fourth tensor should match"
    
    print("All tests passed!")
    
    # Clean up
    shutil.rmtree(folder)


if __name__ == "__main__":
    test_tensor_collection_storage()