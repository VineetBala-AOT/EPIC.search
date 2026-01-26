"""Strategy factory for managing and instantiating search strategies.

This module provides a factory pattern implementation for registering and creating
search strategy instances, with built-in fallback handling and strategy validation.
"""

import logging
from typing import Dict, Type
from .base_strategy import BaseSearchStrategy


class SearchStrategyFactory:
    """Factory class for managing search strategy instances.
    
    This factory uses a registry pattern to manage available search strategies
    and provides safe instantiation with fallback handling for unknown strategies.
    """
    
    # Class-level registry of available strategies
    _strategies: Dict[str, Type[BaseSearchStrategy]] = {}
    _default_strategy: str = "HYBRID_PARALLEL"
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseSearchStrategy]) -> None:
        """Register a search strategy class with the factory.
        
        Args:
            name (str): Strategy name/identifier (e.g., "HYBRID_SEMANTIC_FALLBACK")
            strategy_class (Type[BaseSearchStrategy]): Strategy class that inherits from BaseSearchStrategy
            
        Raises:
            ValueError: If strategy name is invalid or strategy class doesn't inherit from BaseSearchStrategy
        """
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be a non-empty string")
        
        if not issubclass(strategy_class, BaseSearchStrategy):
            raise ValueError("Strategy class must inherit from BaseSearchStrategy")
        
        cls._strategies[name] = strategy_class
        logging.debug(f"Registered search strategy: {name}")
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> BaseSearchStrategy:
        """Get a strategy instance by name, with fallback to default strategy.
        
        Args:
            strategy_name (str): Name of the strategy to instantiate
            
        Returns:
            BaseSearchStrategy: Instance of the requested strategy, or default strategy if not found
        """
        if not strategy_name or not isinstance(strategy_name, str):
            logging.warning(f"Invalid strategy name '{strategy_name}'. Using default strategy '{cls._default_strategy}'")
            strategy_name = cls._default_strategy
        
        strategy_class = cls._strategies.get(strategy_name)
        
        if strategy_class is None:
            logging.warning(f"Unknown search strategy '{strategy_name}'. Using default strategy '{cls._default_strategy}'")
            strategy_class = cls._strategies.get(cls._default_strategy)
            
            if strategy_class is None:
                raise RuntimeError(f"Default strategy '{cls._default_strategy}' is not registered. Ensure all strategies are properly registered.")
        
        try:
            return strategy_class()
        except Exception as e:
            logging.error(f"Failed to instantiate strategy '{strategy_name}': {e}")
            # Try to fallback to default strategy if the requested one fails
            if strategy_name != cls._default_strategy:
                logging.info(f"Attempting fallback to default strategy '{cls._default_strategy}'")
                default_class = cls._strategies.get(cls._default_strategy)
                if default_class:
                    return default_class()
            raise
    
    @classmethod
    def list_strategies(cls) -> Dict[str, str]:
        """List all registered strategies with their descriptions.
        
        Returns:
            dict: Dictionary mapping strategy names to their descriptions
        """
        strategies = {}
        for name, strategy_class in cls._strategies.items():
            try:
                # Create a temporary instance to get the description
                instance = strategy_class()
                strategies[name] = instance.description
            except Exception as e:
                strategies[name] = f"Error getting description: {e}"
        return strategies
    
    @classmethod
    def is_strategy_registered(cls, strategy_name: str) -> bool:
        """Check if a strategy is registered with the factory.
        
        Args:
            strategy_name (str): Name of the strategy to check
            
        Returns:
            bool: True if the strategy is registered, False otherwise
        """
        return strategy_name in cls._strategies
    
    @classmethod
    def set_default_strategy(cls, strategy_name: str) -> None:
        """Set the default fallback strategy.
        
        Args:
            strategy_name (str): Name of the strategy to use as default
            
        Raises:
            ValueError: If the strategy is not registered
        """
        if not cls.is_strategy_registered(strategy_name):
            raise ValueError(f"Cannot set default strategy to '{strategy_name}': strategy is not registered")
        
        cls._default_strategy = strategy_name
        logging.info(f"Default search strategy set to: {strategy_name}")
    
    @classmethod
    def get_default_strategy_name(cls) -> str:
        """Get the name of the current default strategy.
        
        Returns:
            str: Name of the default strategy
        """
        return cls._default_strategy


# Convenience function for external use
def get_search_strategy(strategy_name: str) -> BaseSearchStrategy:
    """Convenience function to get a search strategy instance.
    
    Args:
        strategy_name (str): Name of the strategy to get
        
    Returns:
        BaseSearchStrategy: Strategy instance
    """
    return SearchStrategyFactory.get_strategy(strategy_name)


def list_available_strategies() -> Dict[str, str]:
    """Convenience function to list all available strategies.
    
    Returns:
        dict: Dictionary mapping strategy names to descriptions
    """
    return SearchStrategyFactory.list_strategies()
