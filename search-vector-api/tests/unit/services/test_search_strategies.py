"""Test module for search strategies validation.

This module contains basic tests to ensure the search strategies refactoring
works correctly and all strategies are properly registered.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from services.search_strategies import (
    SearchStrategyFactory,
    get_search_strategy,
    list_available_strategies
)


class TestSearchStrategies(unittest.TestCase):
    """Test cases for search strategy functionality."""
    
    def test_all_strategies_registered(self):
        """Test that all expected strategies are registered with the factory."""
        expected_strategies = [
            "HYBRID_SEMANTIC_FALLBACK",
            "HYBRID_KEYWORD_FALLBACK",
            "SEMANTIC_ONLY",
            "KEYWORD_ONLY", 
            "HYBRID_PARALLEL",
            "DOCUMENT_ONLY"
        ]
        
        available_strategies = list_available_strategies()
        
        for strategy in expected_strategies:
            self.assertIn(strategy, available_strategies, 
                         f"Strategy {strategy} is not registered")
    
    def test_get_strategy_instances(self):
        """Test that we can get strategy instances for all registered strategies."""
        available_strategies = list_available_strategies()
        
        for strategy_name in available_strategies.keys():
            with self.subTest(strategy=strategy_name):
                strategy = get_search_strategy(strategy_name)
                self.assertIsNotNone(strategy)
                self.assertEqual(strategy.strategy_name, strategy_name)
                self.assertIsInstance(strategy.description, str)
                self.assertTrue(len(strategy.description) > 0)
    
    def test_default_strategy_fallback(self):
        """Test that unknown strategies fall back to the default strategy."""
        strategy = get_search_strategy("UNKNOWN_STRATEGY")
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.strategy_name, "HYBRID_PARALLEL")
    
    def test_strategy_validation(self):
        """Test that strategies properly validate their parameters."""
        strategy = get_search_strategy("SEMANTIC_ONLY")
        
        # Test invalid question
        with self.assertRaises(ValueError):
            strategy._validate_parameters("", Mock(), 10, 0.5)
        
        # Test invalid vec_store
        with self.assertRaises(ValueError):
            strategy._validate_parameters("test query", None, 10, 0.5)
        
        # Test invalid top_n
        with self.assertRaises(ValueError):
            strategy._validate_parameters("test query", Mock(), -1, 0.5)
        
        # Test invalid min_relevance_score
        with self.assertRaises(ValueError):
            strategy._validate_parameters("test query", Mock(), 10, -1)
        
        # Test valid parameters (should not raise)
        try:
            strategy._validate_parameters("test query", Mock(), 10, 0.5)
        except ValueError:
            self.fail("Valid parameters should not raise ValueError")
    
    def test_factory_registration(self):
        """Test factory registration functionality."""
        from services.search_strategies.base_strategy import BaseSearchStrategy
        
        class TestStrategy(BaseSearchStrategy):
            @property
            def strategy_name(self):
                return "TEST_STRATEGY"
            
            @property 
            def description(self):
                return "Test strategy"
            
            def execute(self, **kwargs):
                return [], {}
        
        # Register test strategy
        SearchStrategyFactory.register_strategy("TEST_STRATEGY", TestStrategy)
        
        # Verify it's registered
        self.assertTrue(SearchStrategyFactory.is_strategy_registered("TEST_STRATEGY"))
        
        # Verify we can get it
        strategy = get_search_strategy("TEST_STRATEGY")
        self.assertEqual(strategy.strategy_name, "TEST_STRATEGY")
    
    def test_helper_methods(self):
        """Test strategy helper methods."""
        strategy = get_search_strategy("SEMANTIC_ONLY")
        
        # Test time calculation
        import time
        start_time = time.time()
        time.sleep(0.001)  # Sleep for 1ms
        elapsed = strategy._calculate_elapsed_time(start_time)
        self.assertGreater(elapsed, 0)
        self.assertIsInstance(elapsed, float)
        
        # Test metrics update
        metrics = {}
        strategy._update_metrics(metrics, test_key="test_value", test_number=123)
        self.assertEqual(metrics["test_key"], "test_value")
        self.assertEqual(metrics["test_number"], 123)


if __name__ == '__main__':
    unittest.main()
