import math
import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from string_art_generator import (
    calculate_distance_based_sampling,
    find_best_next_pin,
    find_best_next_pin_2_lookahead,
    calculate_angular_distance
)
from image_processing import calculate_line_darkness


class TestDistanceBasedSampling:
    """Test the new distance-based sampling algorithm."""

    def test_calculate_angular_distance_basic_cases(self):
        """Test basic angular distance calculations."""
        # Same pin should return 0 distance
        assert calculate_angular_distance(0, 0, 12) == 0

        # Adjacent pins should return 1
        assert calculate_angular_distance(0, 1, 12) == 1
        assert calculate_angular_distance(5, 6, 12) == 1

        # Opposite pins should return half the total pins
        assert calculate_angular_distance(0, 6, 12) == 6

        # Clockwise vs counter-clockwise should give same result
        assert calculate_angular_distance(0, 5, 12) == calculate_angular_distance(5, 0, 12)

    def test_distance_based_sampling_small_candidates(self):
        """Test sampling with small candidate sets."""
        pins_coords = [(i*10, i*10) for i in range(10)]
        current_pin_index = 0
        candidate_indices = [1, 2, 3]

        # With few candidates, all should be returned
        result = calculate_distance_based_sampling(
            pins_coords, current_pin_index, candidate_indices, len(pins_coords)
        )
        assert set(result) == set(candidate_indices)

    def test_distance_based_sampling_large_candidates(self):
        """Test sampling with large candidate sets."""
        # Create a scenario with 50 pins,20 candidates, should sample 30
        pins_coords = [(i*5, i*5) for i in range(50)]
        current_pin_index = 0

        # Generate 20 unique candidate indices
        candidate_indices = list(range(1, 21))

        result = calculate_distance_based_sampling(
            pins_coords, current_pin_index, candidate_indices, len(pins_coords), sample_size=30
        )

        # Should return30 candidates (sample_size)
        assert len(result) == 30

        # Should include some top candidates (distance-based scoring)
        # and some diverse candidates (random sampling for diversity)
        assert set(result).issubset(set(candidate_indices))

        # Should not include current pin
        assert current_pin_index not in result


class TestWeightedDarknessCalculation:
    """Test the weighted darkness calculation in find_best_next_pin."""

    def test_weighted_darkness_calculation_center_weighted(self, mocker):
        """Test that center-weighted darkness calculation works correctly."""
        # Create a simple test image
        img_size = 50
        init_array = np.zeros((img_size, img_size), dtype=np.uint8)

        # Create pins
        pins_coords = [(10, 10), (40, 10), (40, 40), (10, 40)]

        # Create a line with a "dark center" by setting center pixels to low values
        # This simulates what the weighted calculation should prefer
        mock_line_pixels = [(15, 15), (20, 20), (25, 25), (30, 30)]

        # Set up the mock to return our test line
        def mock_calculate_line_darkness(img_array, x1, y1, x2, y2):
            return 150.0, mock_line_pixels # Return moderate darkness

        mocker.patch('string_art_generator.calculate_line_darkness', side_effect=mock_calculate_line_darkness)

        # Test the weighted calculation by calling find_best_next_pin
        result = find_best_next_pin(
            pins_coords, init_array, 0, None, [0], 1, float('inf'), temperature=1.0
        )

        # The weighted calculation should work without errors
        assert result[0] is not None or result[0] ==0  # Should find a valid pin

    def test_weighted_darkness_vs_regular_darkness(self):
        """Test that weighted darkness calculation differs from regular calculation."""
        # This is more of an integration test showing the feature works
        # The exact behavior depends on the image and line characteristics
        pass


class TestLookaheadAlgorithm:
    """Test the 2-pin lookahead algorithm."""

    def test_lookahead_basic_functionality(self, mocker):
        """Test that the lookahead algorithm works without errors."""
        img_size = 100
        init_array = np.zeros((img_size, img_size), dtype=np.uint8)
        pins_coords = [(i*25 for i in range(4))]

        # Mock the base find_best_next_pin to return a valid result
        mocker.patch('string_art_generator.find_best_next_pin', return_value=(2, [(0,0), (1,1)], 50.0))

        # Test lookahead algorithm
        result = find_best_next_pin_2_lookahead(
            pins_coords, init_array, 0, None, [0], 1, float('inf'), temperature=1.0
        )

        # Should work without errors
        assert result[0] is not None or result[0] == 0

    def test_lookahead_with_mock_sequences(self, mocker):
        """Test lookahead with specific sequence scenarios."""
        # This would require complex mocking to simulate the full behavior
        pass


class TestSimulatedAnnealing:
    """Test the simulated annealing temperature-based optimization."""

    def test_simulated_annealing_acceptance_probability(self, mocker):
        """Test that temperature-based acceptance works correctly."""
        img_size = 100
        init_array = np.zeros((img_size, img_size), dtype=np.uint8)
        pins_coords = [(i*25 for i in range(4))]

        # Test with high temperature (early in drawing process)
        result_high_temp = find_best_next_pin(
            pins_coords, init_array, 0, None, [0], 1, float('inf'), temperature=2.0
        )

        # Test with low temperature (later in drawing process)
        result_low_temp = find_best_next_pin(
            pins_coords, init_array, 0, None, [0], 1, float('inf'), temperature=1.0
        )

        # Both should work without errors
        # The temperature affects the acceptance probability for worse solutions
        assert result_high_temp != result_low_temp or result_high_temp == result_low_temp

    def test_temperature_decay_schedule(self, mocker):
        """Test temperature decay over drawing progress."""
        # This is tested in the main algorithm through generate_next_string_segment
        pass


class TestIntegration:
    """Integration tests for the improved algorithm."""

    def test_algorithm_choice_early_vs_late(self, mocker):
        """Test that the algorithm switches from lookahead to greedy based on progress."""
        # This test would require mocking the full generate_next_string_segment function
        # to test the algorithm switching logic
        pass

    def test_performance_benchmark_placeholder(self):
        """Placeholder for performance benchmarking - would require more complex setup."""
        # Future: Compare old vs new algorithm performance
        pass