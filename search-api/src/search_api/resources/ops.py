# Copyright © 2019 Province of British Columbia
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Endpoints to check and manage the health of the service."""
from flask import current_app, request
from flask_restx import Namespace, Resource

# from api.models import db
from ..utils.run_version import get_run_version
from ..auth import auth


API = Namespace('', description='Service - OPS checks')

@API.route('healthz')
class Healthz(Resource):
    """Determines if the service and required dependencies are still working.

    This could be thought of as a heartbeat for the service.
    """

    @staticmethod
    def get():
        """Return a JSON object stating the health of the Service and dependencies."""
        current_app.logger.info("Health check endpoint called")
        
        # Could add more detailed health checks here in the future:
        # - Database connectivity
        # - External service availability
        # - Memory/disk usage
        
        current_app.logger.info("Health check completed successfully")
        # made it here, so all checks passed
        return {'message': 'api is healthy'}, 200


@API.route('readyz')
class Readyz(Resource):
    """Determines if the service is ready to respond."""

    @staticmethod
    def get():
        """Return a JSON object that identifies if the service is setupAnd ready to work."""
        current_app.logger.info("Readiness check endpoint called")
        
        # Could add more detailed readiness checks here in the future:
        # - Configuration validation
        # - Required services connectivity
        # - Initialization completion
        
        current_app.logger.info("Readiness check completed successfully")
        return {'message': 'api is ready'}, 200


@API.route('version')
class Version(Resource):
    """Expose the running application version for deployment tracking."""

    @staticmethod
    def get():
        """Return the current release identifier."""
        current_app.logger.info("Version endpoint called")
        return {
            'version': get_run_version()
        }, 200


@API.route('cache-status')
class CacheStatus(Resource):
    """Cache monitoring and management endpoint."""

    @staticmethod
    def get():
        """Return cache statistics and status information."""
        current_app.logger.info("Cache status endpoint called")
        
        try:
            from ..utils.cache import get_cache_stats
            stats = get_cache_stats()
            
            # Add human readable information
            cache_info = {
                'cache_statistics': stats,
                'cache_health': {
                    'total_entries': stats['total_entries'],
                    'expired_entries': stats['expired_entries'],
                    'active_entries': stats['total_entries'] - stats['expired_entries'],
                    'cache_hit_potential': f"{((stats['total_entries'] - stats['expired_entries']) / max(stats['total_entries'], 1)) * 100:.1f}%"
                },
                'cache_keys_summary': [
                    {
                        'function': key_info['key'].split(':')[0] if ':' in key_info['key'] else key_info['key'],
                        'age_hours': round(key_info['age_seconds'] / 3600, 1),
                        'is_expired': key_info['is_expired']
                    }
                    for key_info in stats['cache_keys']
                ]
            }
            
            current_app.logger.info(f"Cache status: {stats['total_entries']} total, {stats['expired_entries']} expired")
            return cache_info, 200
            
        except Exception as e:
            current_app.logger.error(f"Error getting cache status: {e}")
            return {'error': 'Failed to get cache status', 'message': str(e)}, 500

    @staticmethod
    def delete():
        """Clear expired cache entries."""
        current_app.logger.info("Cache cleanup endpoint called")

        try:
            from ..utils.cache import clear_expired_cache
            cleared_count = clear_expired_cache()

            current_app.logger.info(f"Cache cleanup completed: {cleared_count} expired entries removed")
            return {
                'message': 'Cache cleanup completed',
                'expired_entries_removed': cleared_count
            }, 200

        except Exception as e:
            current_app.logger.error(f"Error clearing cache: {e}")
            return {'error': 'Failed to clear cache', 'message': str(e)}, 500

    @staticmethod
    @auth.requires_epic_search_role(["admin"])
    def post():
        """Force-refresh project metadata and document type caches (admin only).

        Call this after EPIC updates project phase/status/proponent so search
        picks up the new values within seconds rather than waiting up to 30 min.

        Optional JSON body: {"target": "projects"} or {"target": "document_types"}
        to clear only one cache.  Omit or use {"target": "all"} to clear both.
        """
        current_app.logger.info("Cache invalidation endpoint called")

        try:
            from ..utils.cache import clear_cache_by_prefix, clear_cache

            body = request.get_json(silent=True) or {}
            target = body.get("target", "all")

            removed = {}
            if target in ("all", "projects"):
                removed["projects"] = clear_cache_by_prefix("get_projects_list")
                current_app.logger.info(f"Cleared {removed['projects']} project cache entries")

            if target in ("all", "document_types"):
                removed["document_types"] = clear_cache_by_prefix("get_document_types")
                current_app.logger.info(f"Cleared {removed['document_types']} document_type cache entries")

            total = sum(removed.values())
            current_app.logger.info(f"Cache invalidation complete: {total} entries removed ({removed})")
            return {
                "message": "Cache cleared — next query will fetch fresh data from the database",
                "entries_removed": removed,
            }, 200

        except Exception as e:
            current_app.logger.error(f"Error clearing cache: {e}")
            return {"error": "Failed to clear cache", "message": str(e)}, 500
