import unittest
import torch
from fbfgs.sparse_flat_tensor import SparseFlatTensor

class TestSparseFlatTensor(unittest.TestCase):

    def _dense_to_sparse_flat_tensor_components(self, dense_tensor: torch.Tensor):
        """
        Helper to get the components for SparseFlatTensor from a dense tensor.
        This mirrors the logic in dense_to_sparse_flat_tensor for testing purposes.
        """
        device = dense_tensor.device
        dtype = dense_tensor.dtype
        total_size = dense_tensor.numel()

        non_zero_indices = torch.nonzero(dense_tensor.view(-1)).squeeze()

        if non_zero_indices.numel() == total_size:
            return {
                "starts": torch.tensor([0], dtype=torch.int64, device=device),
                "ends": torch.tensor([total_size], dtype=torch.int64, device=device),
                "values": dense_tensor.view(-1),
                "total_size": total_size,
                "unit_indices": torch.empty(0, dtype=torch.long, device=device),
                "unit_values": torch.empty(0, dtype=dtype, device=device)
            }
        
        if non_zero_indices.numel() == 0:
            return {
                "starts": torch.empty(0, dtype=torch.int64, device=device),
                "ends": torch.empty(0, dtype=torch.int64, device=device),
                "values": torch.empty(0, dtype=dtype, device=device),
                "total_size": total_size,
                "unit_indices": torch.empty(0, dtype=torch.long, device=device),
                "unit_values": torch.empty(0, dtype=dtype, device=device)
            }

        diff = non_zero_indices[1:] - non_zero_indices[:-1]
        segment_ends_indices = torch.nonzero(diff > 1).squeeze() + 1
        if segment_ends_indices.ndim == 0 and segment_ends_indices.numel() > 0:
            segment_ends_indices = segment_ends_indices.unsqueeze(0)
        elif segment_ends_indices.numel() == 0:
            segment_ends_indices = torch.empty(0, dtype=torch.long, device=device)

        start_indices_in_non_zero = torch.cat([torch.tensor([0], dtype=torch.long, device=device), segment_ends_indices])
        end_indices_in_non_zero = torch.cat([segment_ends_indices - 1, torch.tensor([len(non_zero_indices) - 1], dtype=torch.long, device=device)])

        starts_local_segments = non_zero_indices[start_indices_in_non_zero]
        ends_local_segments = non_zero_indices[end_indices_in_non_zero] + 1
        segment_lengths = ends_local_segments - starts_local_segments

        is_unit_segment = (segment_lengths == 1)
        
        unit_indices_local = starts_local_segments[is_unit_segment]
        unit_values_local = dense_tensor.view(-1)[unit_indices_local]

        segment_mask = ~is_unit_segment
        starts_local = starts_local_segments[segment_mask]
        ends_local = ends_local_segments[segment_mask]
        
        if segment_mask.any():
            seg_starts = start_indices_in_non_zero[segment_mask]
            seg_ends = end_indices_in_non_zero[segment_mask]
            
            segment_lengths_vals = seg_ends - seg_starts + 1
            total_segment_elements = segment_lengths_vals.sum()
            
            segment_start_repeated = torch.repeat_interleave(seg_starts, segment_lengths_vals)
            intra_offsets = torch.arange(total_segment_elements, device=device) - torch.repeat_interleave(
                torch.cat([torch.tensor([0], device=device), segment_lengths_vals.cumsum(0)[:-1]]), 
                segment_lengths_vals
            )
            
            segment_indices = non_zero_indices[segment_start_repeated + intra_offsets]
            values_local = dense_tensor.view(-1)[segment_indices]
        else:
            values_local = torch.empty(0, dtype=dtype, device=device)

        return {
            "starts": starts_local,
            "ends": ends_local,
            "values": values_local,
            "total_size": total_size,
            "unit_indices": unit_indices_local,
            "unit_values": unit_values_local
        }

    def test_pin_memory(self):
        if not torch.cuda.is_available():
            # Test on CPU
            dense_original = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0], device='cpu')
            sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)

            # Ensure original is not pinned
            self.assertFalse(sparse_flat.starts.is_pinned())
            self.assertFalse(sparse_flat.ends.is_pinned())
            self.assertFalse(sparse_flat.values.is_pinned())
            self.assertFalse(sparse_flat.total_size.is_pinned())
            self.assertFalse(sparse_flat.unit_indices.is_pinned())
            self.assertFalse(sparse_flat.unit_values.is_pinned())

            pinned_sparse_flat = sparse_flat.pin_memory()

            # Ensure new object is returned
            self.assertIsNot(sparse_flat, pinned_sparse_flat)

            # Ensure all tensors in the new object are pinned
            self.assertTrue(pinned_sparse_flat.starts.is_pinned())
            self.assertTrue(pinned_sparse_flat.ends.is_pinned())
            self.assertTrue(pinned_sparse_flat.values.is_pinned())
            self.assertTrue(pinned_sparse_flat.total_size.is_pinned())
            self.assertTrue(pinned_sparse_flat.unit_indices.is_pinned())
            self.assertTrue(pinned_sparse_flat.unit_values.is_pinned())

            # Ensure original object is unchanged
            self.assertFalse(sparse_flat.starts.is_pinned())
            self.assertFalse(sparse_flat.ends.is_pinned())
            self.assertFalse(sparse_flat.values.is_pinned())
            self.assertFalse(sparse_flat.total_size.is_pinned())
            self.assertFalse(sparse_flat.unit_indices.is_pinned())
            self.assertFalse(sparse_flat.unit_values.is_pinned())
        else:
            # Test on CUDA, then pin and move to CPU (not applicable directly, pin_memory is for CPU->CUDA transfer)
            # The primary use case for pin_memory() is to speed up host-to-device transfers.
            # If a tensor is already on CUDA, pin_memory() does nothing, or raises an error.
            # So, we test pinning on CPU only, and ensure it remains on CPU.
            dense_original = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0], device='cpu')
            sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
            pinned_sparse_flat = sparse_flat.pin_memory()
            self.assertTrue(pinned_sparse_flat.values.is_pinned())
            self.assertEqual(pinned_sparse_flat.values.device.type, 'cpu')

    def test_to_method(self):
        dense_original = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0], device='cpu')
        sparse_flat_cpu = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)

        # Test moving to CPU (even if already on CPU, it should return a new object)
        sparse_flat_cpu_new = sparse_flat_cpu.to(device='cpu')
        self.assertIsNot(sparse_flat_cpu, sparse_flat_cpu_new)
        self.assertEqual(sparse_flat_cpu_new.values.device.type, 'cpu')
        torch.testing.assert_close(sparse_flat_cpu.to_dense(), sparse_flat_cpu_new.to_dense())

        if torch.cuda.is_available():
            # Test moving to CUDA
            sparse_flat_cuda = sparse_flat_cpu.to(device='cuda')

            # Ensure new object is returned and on CUDA
            self.assertIsNot(sparse_flat_cpu, sparse_flat_cuda)
            self.assertEqual(sparse_flat_cuda.values.device.type, 'cuda')

            # Ensure values are preserved
            torch.testing.assert_close(sparse_flat_cpu.to_dense(), sparse_flat_cuda.to_dense().to(device='cpu'))

            # Test moving from CUDA back to CPU
            sparse_flat_cuda_to_cpu = sparse_flat_cuda.to(device='cpu')
            self.assertIsNot(sparse_flat_cuda, sparse_flat_cuda_to_cpu)
            self.assertEqual(sparse_flat_cuda_to_cpu.values.device.type, 'cpu')
            torch.testing.assert_close(sparse_flat_cuda.to_dense().to(device='cpu'), sparse_flat_cuda_to_cpu.to_dense())

            # Test with non_blocking=True
            sparse_flat_cuda_non_blocking = sparse_flat_cpu.to(device='cuda', non_blocking=True)
            self.assertEqual(sparse_flat_cuda_non_blocking.values.device.type, 'cuda')
            torch.testing.assert_close(sparse_flat_cpu.to_dense(), sparse_flat_cuda_non_blocking.to_dense().to(device='cpu'))
        else:
            print("CUDA not available, skipping CUDA device tests for .to() method.")

    def test_dot_sparse_flat_tensor(self):
        # Test case 1: Basic dot product
        dense_a = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0])
        dense_b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        sparse_a = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_a)
        sparse_b = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_b)

        result_sparse_dot = sparse_a.dot(sparse_b)
        expected_dense_dot = torch.dot(dense_a, dense_b)

        torch.testing.assert_close(result_sparse_dot, expected_dense_dot)

        # Test case 2: Dot product with one all-zero tensor
        dense_a = torch.tensor([0.0, 0.0, 0.0])
        dense_b = torch.tensor([1.0, 2.0, 3.0])

        sparse_a = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_a)
        sparse_b = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_b)

        result_sparse_dot = sparse_a.dot(sparse_b)
        expected_dense_dot = torch.dot(dense_a, dense_b)
        torch.testing.assert_close(result_sparse_dot, expected_dense_dot)

        # Test case 3: Dot product with non-overlapping non-zero elements
        dense_a = torch.tensor([1.0, 0.0, 0.0, 2.0])
        dense_b = torch.tensor([0.0, 3.0, 4.0, 0.0])

        sparse_a = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_a)
        sparse_b = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_b)

        result_sparse_dot = sparse_a.dot(sparse_b)
        expected_dense_dot = torch.dot(dense_a, dense_b)
        torch.testing.assert_close(result_sparse_dot, expected_dense_dot)






    def test_dense_to_sparse_to_dense_roundtrip(self):
        # Test case 1: Basic tensor with some zeros and unit values
        dense_original = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        dense_reconstructed = sparse_flat.to_dense()
        torch.testing.assert_close(dense_original, dense_reconstructed)

        # Test case 2: All zeros
        dense_original = torch.zeros(10)
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        dense_reconstructed = sparse_flat.to_dense()
        torch.testing.assert_close(dense_original, dense_reconstructed)

        # Test case 3: All non-zeros
        dense_original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        dense_reconstructed = sparse_flat.to_dense()
        torch.testing.assert_close(dense_original, dense_reconstructed)

        # Test case 4: Mixed segments and unit values, at boundaries
        dense_original = torch.tensor([1.0, 0.0, 2.0, 3.0, 0.0, 4.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        dense_reconstructed = sparse_flat.to_dense()
        torch.testing.assert_close(dense_original, dense_reconstructed)

        # Test case 5: Empty tensor
        dense_original = torch.empty(0)
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        dense_reconstructed = sparse_flat.to_dense()
        torch.testing.assert_close(dense_original, dense_reconstructed)


    def test_mul_scalar(self):
        scalar = 5.0
        
        # Test case 1: Segments only
        dense_original = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)

        sparse_multiplied = sparse_flat * scalar
        
        dense_expected = dense_original * scalar
        dense_reconstructed = sparse_multiplied.to_dense()
        torch.testing.assert_close(dense_expected, dense_reconstructed)

        # Test case 2: Unit values only
        dense_original = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        sparse_multiplied = sparse_flat * scalar
        dense_expected = dense_original * scalar
        dense_reconstructed = sparse_multiplied.to_dense()
        torch.testing.assert_close(dense_expected, dense_reconstructed)

        # Test case 3: Mixed
        dense_original = torch.tensor([1.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        sparse_multiplied = sparse_flat * scalar
        dense_expected = dense_original * scalar
        dense_reconstructed = sparse_multiplied.to_dense()
        torch.testing.assert_close(dense_expected, dense_reconstructed)


    def test_add_sparse_dense(self):
        # Test case 1: Unit values only
        sparse_val_dense = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0])
        dense_original = torch.ones(8) * 10
        
        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        sparse_flat = SparseFlatTensor(**components)

        result_dense = SparseFlatTensor.add_sparse_dense(sparse_flat, dense_original.clone()) # clone to avoid in-place modification for comparison
        expected_dense = dense_original + sparse_val_dense
        
        torch.testing.assert_close(result_dense, expected_dense)

        # Test case 2: Segments only
        sparse_val_dense = torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0])
        dense_original = torch.ones(8) * 10

        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        sparse_flat = SparseFlatTensor(**components)

        result_dense = SparseFlatTensor.add_sparse_dense(sparse_flat, dense_original.clone())
        expected_dense = dense_original + sparse_val_dense
        torch.testing.assert_close(result_dense, expected_dense)

        # Test case 3: Mixed
        sparse_val_dense = torch.tensor([1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 6.0])
        dense_original = torch.ones(9) * 10
        
        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        sparse_flat = SparseFlatTensor(**components)

        result_dense = SparseFlatTensor.add_sparse_dense(sparse_flat, dense_original.clone())
        expected_dense = dense_original + sparse_val_dense
        torch.testing.assert_close(result_dense, expected_dense)

        # Test case 4: Offset addition (using _add_sparse_dense for this)
        # Assuming _add_sparse_dense has similar logic but with offset
        sparse_val_dense = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 0.0]) # Total size 6
        dense_sub_tensor = torch.zeros(4) # Dense tensor of size 4
        offset = 1 # sparse_val_dense[1:5] should map to dense_sub_tensor[0:4]

        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        # Create SparseFlatTensor that represents the full sparse_val_dense
        full_sparse_flat = SparseFlatTensor(**components) 

        # Manually extract the relevant part of sparse_val_dense for the offset region
        expected_sub_tensor = dense_sub_tensor.clone()
        expected_sub_tensor += sparse_val_dense[offset : offset + dense_sub_tensor.numel()]

        # Call the actual method with offset
        # Note: _add_sparse_dense modifies in-place, so clone dense_sub_tensor
        result_sub_tensor = SparseFlatTensor._add_sparse_dense_alpha(full_sparse_flat, dense_sub_tensor.clone(), offset=offset)
        
        torch.testing.assert_close(result_sub_tensor, expected_sub_tensor)

        # Test case 5: _add_sparse_dense_alpha with alpha and offset
        sparse_val_dense = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 0.0]) # Total size 6
        dense_sub_tensor = torch.zeros(4) # Dense tensor of size 4
        offset = 1 
        alpha = 2.0

        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        full_sparse_flat = SparseFlatTensor(**components) 

        expected_sub_tensor = dense_sub_tensor.clone()
        expected_sub_tensor += alpha * sparse_val_dense[offset : offset + dense_sub_tensor.numel()]

        result_sub_tensor = SparseFlatTensor._add_sparse_dense_alpha(full_sparse_flat, dense_sub_tensor.clone(), alpha=alpha, offset=offset)
        
        torch.testing.assert_close(result_sub_tensor, expected_sub_tensor)


    def test_sparse_dot_dense(self):
        # Test case 1: Unit values only
        sparse_val_dense = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0])
        dense_other = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        sparse_flat = SparseFlatTensor(**components)

        result_dot = SparseFlatTensor.sparse_dot_dense(sparse_flat, dense_other)
        expected_dot = torch.dot(sparse_val_dense, dense_other)
        
        torch.testing.assert_close(result_dot, expected_dot)

        # Test case 2: Segments only
        sparse_val_dense = torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0])
        dense_other = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        sparse_flat = SparseFlatTensor(**components)

        result_dot = SparseFlatTensor.sparse_dot_dense(sparse_flat, dense_other)
        expected_dot = torch.dot(sparse_val_dense, dense_other)
        torch.testing.assert_close(result_dot, expected_dot)

        # Test case 3: Mixed
        sparse_val_dense = torch.tensor([1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 6.0])
        dense_other = torch.arange(9, dtype=torch.float32) # [0, 1, ..., 8]

        components = self._dense_to_sparse_flat_tensor_components(sparse_val_dense)
        sparse_flat = SparseFlatTensor(**components)

        result_dot = SparseFlatTensor.sparse_dot_dense(sparse_flat, dense_other)
        expected_dot = torch.dot(sparse_val_dense, dense_other)
        torch.testing.assert_close(result_dot, expected_dot)


    def test_rmul_with_dense_tensor(self):
        # Test case 1: Basic element-wise multiplication
        dense_original = torch.tensor([1.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0])
        other_dense = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        
        result_tensor = other_dense * sparse_flat
        expected_tensor = other_dense * dense_original
        
        torch.testing.assert_close(result_tensor, expected_tensor)

        # Test case 2: Different types of sparsity
        dense_original = torch.tensor([0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
        other_dense = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)

        result_tensor = other_dense * sparse_flat
        expected_tensor = other_dense * dense_original
        
        torch.testing.assert_close(result_tensor, expected_tensor)

        # Test case 3: All zeros sparse tensor
        dense_original = torch.zeros(5)
        other_dense = torch.arange(5, dtype=torch.float32)

        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)

        result_tensor = other_dense * sparse_flat
        expected_tensor = other_dense * dense_original

        torch.testing.assert_close(result_tensor, expected_tensor)

        # Test case 4: All non-zeros sparse tensor
        dense_original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        other_dense = torch.full((5,), 2.0)

        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)

        result_tensor = other_dense * sparse_flat
        expected_tensor = other_dense * dense_original

        torch.testing.assert_close(result_tensor, expected_tensor)


    def test_get_nonzero_mask(self):
        # Test case 1: Basic mixed tensor
        dense_original = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        
        expected_mask = (dense_original != 0)
        actual_mask = sparse_flat.get_nonzero_mask()
        
        torch.testing.assert_close(actual_mask, expected_mask)

        # Test case 2: All zeros
        dense_original = torch.zeros(7)
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        
        expected_mask = (dense_original != 0)
        actual_mask = sparse_flat.get_nonzero_mask()
        
        torch.testing.assert_close(actual_mask, expected_mask)

        # Test case 3: All non-zeros
        dense_original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        
        expected_mask = (dense_original != 0)
        actual_mask = sparse_flat.get_nonzero_mask()
        
        torch.testing.assert_close(actual_mask, expected_mask)

        # Test case 4: Empty tensor
        dense_original = torch.empty(0)
        sparse_flat = SparseFlatTensor.dense_to_sparse_flat_tensor(dense_original)
        
        expected_mask = (dense_original != 0)
        actual_mask = sparse_flat.get_nonzero_mask()
        
        torch.testing.assert_close(actual_mask, expected_mask)


if __name__ == '__main__':
    unittest.main()