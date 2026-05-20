# Copyright © 2024 Province of British Columbia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bring in the common cache."""
from flask_caching import Cache
import time
import functools
from typing import Callable, Any

# lower case name as used by convention in most Flask apps
cache = Cache(config={'CACHE_TYPE': 'simple'})  # pylint: disable=invalid-name

# Simple in-memory cache with TTL for API responses
_memory_cache = {}

def cache_with_ttl(ttl_seconds: int = 3600):
    """
    Simple TTL cache decorator for API responses.
    
    Args:
        ttl_seconds (int): Time to live in seconds (default: 1 hour)
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check if we have a cached value that hasn't expired
            if cache_key in _memory_cache:
                cached_data, timestamp = _memory_cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return cached_data
                else:
                    # Remove expired entry
                    del _memory_cache[cache_key]
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            _memory_cache[cache_key] = (result, time.time())
            
            return result
        return wrapper
    return decorator


def get_cache_stats():
    """Get cache statistics for monitoring."""
    current_time = time.time()
    stats = {
        'total_entries': len(_memory_cache),
        'expired_entries': 0,
        'cache_keys': []
    }
    
    for key, (data, timestamp) in _memory_cache.items():
        stats['cache_keys'].append({
            'key': key,
            'age_seconds': int(current_time - timestamp),
            'is_expired': current_time - timestamp > 86400  # Default 24h TTL check
        })
        if current_time - timestamp > 86400:
            stats['expired_entries'] += 1
    
    return stats


def clear_cache():
    """Clear all cached entries."""
    global _memory_cache
    _memory_cache.clear()


def clear_cache_by_prefix(prefix: str) -> int:
    """Clear all cache entries whose key starts with the given function name prefix.

    Args:
        prefix: Function name to match (e.g. "get_projects_list")

    Returns:
        Number of entries removed
    """
    global _memory_cache
    keys_to_remove = [k for k in _memory_cache if k.startswith(prefix)]
    for k in keys_to_remove:
        del _memory_cache[k]
    return len(keys_to_remove)


def clear_expired_cache():
    """Clear only expired cache entries."""
    global _memory_cache
    current_time = time.time()
    expired_keys = []
    
    for key, (data, timestamp) in _memory_cache.items():
        if current_time - timestamp > 86400:  # Default 24h TTL
            expired_keys.append(key)
    
    for key in expired_keys:
        del _memory_cache[key]
    
    return len(expired_keys)