"""Enhanced Agent Wrapper - Integrates enhanced capabilities with existing VectorSearchAgent.

This wrapper adds Phase 1 enhancements (planning, self-evaluation, refinement)
while maintaining compatibility with the existing agent_stub.py architecture.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from .enhanced_agent import (
    EnhancedAgentPlanner,
    AgentSelfEvaluator,
    AgentRefiner,
    EvaluationResult,
    PlanningResult
)

logger = logging.getLogger(__name__)


class EnhancedVectorSearchAgent:
    """Enhanced wrapper around VectorSearchAgent with planning, evaluation, and refinement."""

    def __init__(self, base_agent, llm_client):
        """Initialize enhanced agent wrapper.

        Args:
            base_agent: Existing VectorSearchAgent instance
            llm_client: LLM client for enhanced capabilities
        """
        self.base_agent = base_agent
        self.llm_client = llm_client

        # Enhanced modules
        self.planner = EnhancedAgentPlanner(llm_client)
        self.evaluator = AgentSelfEvaluator(llm_client)
        self.refiner = AgentRefiner(llm_client)

        # Execution tracking
        self.execution_log = []
        self.refinement_count = 0
        self.max_refinements = 1  # Limit refinement iterations

    def execute_with_enhancements(self, query: str, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query with enhanced planning, evaluation, and refinement.

        Args:
            query: User query
            reason: Reason for agent mode selection
            context: Execution context

        Returns:
            Enhanced results with evaluation metadata
        """
        logger.info("🚀 ENHANCED AGENT: Starting enhanced execution pipeline...")
        start_time = time.time()

        # PHASE 1: Enhanced Planning
        logger.info("🚀 PHASE 1: Creating enhanced execution plan...")
        planning_start = time.time()

        try:
            planning_result = self.planner.create_enhanced_plan(query, reason, context)
            planning_time = (time.time() - planning_start) * 1000

            self._log_step({
                "phase": "planning",
                "duration_ms": planning_time,
                "complexity": planning_result.estimated_complexity,
                "steps_planned": len(planning_result.execution_plan),
                "reasoning": planning_result.reasoning[:200]
            })

            logger.info(f"🚀 PHASE 1 COMPLETE: Plan created with {len(planning_result.execution_plan)} steps "
                       f"(complexity: {planning_result.estimated_complexity}) in {planning_time:.0f}ms")

        except Exception as e:
            logger.error(f"🚀 PHASE 1 ERROR: Planning failed: {e} - falling back to base agent planning")
            # Fallback to base agent's create_execution_plan
            planning_result = None
            planning_time = 0

        # PHASE 2: Execute Plan (using base agent or enhanced plan)
        logger.info("🚀 PHASE 2: Executing search plan...")
        execution_start = time.time()

        try:
            if planning_result:
                # Use enhanced plan
                execution_result = self.base_agent.execute_plan(
                    planning_result.execution_plan,
                    context
                )
            else:
                # Fallback: let base agent create and execute its own plan
                base_plan = self.base_agent.create_execution_plan(query, reason)
                execution_result = self.base_agent.execute_plan(base_plan, context)

            execution_time = (time.time() - execution_start) * 1000

            self._log_step({
                "phase": "execution",
                "duration_ms": execution_time,
                "documents_found": len(execution_result.get('documents', [])),
                "chunks_found": len(execution_result.get('document_chunks', []))
            })

            logger.info(f"🚀 PHASE 2 COMPLETE: Execution finished in {execution_time:.0f}ms - "
                       f"found {len(execution_result.get('documents', []))} documents, "
                       f"{len(execution_result.get('document_chunks', []))} chunks")

        except Exception as e:
            logger.error(f"🚀 PHASE 2 ERROR: Execution failed: {e}")
            # Return error result
            return {
                "error": True,
                "message": f"Execution failed: {str(e)}",
                "execution_log": self.execution_log,
                "phase_failed": "execution"
            }

        # PHASE 3: Self-Evaluation
        logger.info("🚀 PHASE 3: Evaluating result quality...")
        evaluation_start = time.time()

        try:
            evaluation = self.evaluator.evaluate_results(
                query,
                execution_result,
                {"execution_time_ms": execution_time}
            )

            evaluation_time = (time.time() - evaluation_start) * 1000

            self._log_step({
                "phase": "evaluation",
                "duration_ms": evaluation_time,
                "quality_score": evaluation.quality_score,
                "relevance_score": evaluation.relevance_score,
                "coverage_score": evaluation.coverage_score,
                "needs_refinement": evaluation.needs_refinement,
                "reasoning": evaluation.reasoning[:200]
            })

            logger.info(f"🚀 PHASE 3 COMPLETE: Evaluation finished in {evaluation_time:.0f}ms - "
                       f"quality: {evaluation.quality_score:.2f}, "
                       f"needs refinement: {evaluation.needs_refinement}")

        except Exception as e:
            logger.error(f"🚀 PHASE 3 ERROR: Evaluation failed: {e} - skipping refinement")
            evaluation = None
            evaluation_time = 0

        # PHASE 4: Refinement (if needed)
        refinement_executed = False
        refinement_time = 0

        if evaluation and evaluation.needs_refinement and self.refinement_count < self.max_refinements:
            logger.info("🚀 PHASE 4: Quality below threshold - starting refinement...")
            refinement_start = time.time()

            try:
                refinement_plan = self.refiner.create_refinement_plan(
                    query,
                    execution_result,
                    evaluation
                )

                if refinement_plan:
                    logger.info(f"🚀 PHASE 4: Executing {len(refinement_plan)} refinement steps...")

                    # Execute refinement steps
                    refinement_result = self.base_agent.execute_plan(
                        refinement_plan,
                        context
                    )

                    # Merge refinement results with original results
                    execution_result = self._merge_results(execution_result, refinement_result)

                    # Re-evaluate after refinement
                    evaluation = self.evaluator.evaluate_results(
                        query,
                        execution_result,
                        {"execution_time_ms": execution_time}
                    )

                    refinement_executed = True
                    self.refinement_count += 1

                refinement_time = (time.time() - refinement_start) * 1000

                self._log_step({
                    "phase": "refinement",
                    "duration_ms": refinement_time,
                    "steps_executed": len(refinement_plan) if refinement_plan else 0,
                    "new_quality_score": evaluation.quality_score if evaluation else None,
                    "improvement": True if refinement_executed else False
                })

                logger.info(f"🚀 PHASE 4 COMPLETE: Refinement finished in {refinement_time:.0f}ms - "
                           f"new quality: {evaluation.quality_score:.2f if evaluation else 'N/A'}")

            except Exception as e:
                logger.error(f"🚀 PHASE 4 ERROR: Refinement failed: {e}")
                refinement_time = 0

        elif evaluation and not evaluation.needs_refinement:
            logger.info(f"🚀 PHASE 4 SKIPPED: Quality score {evaluation.quality_score:.2f} "
                       f"meets threshold - no refinement needed")
        elif self.refinement_count >= self.max_refinements:
            logger.info(f"🚀 PHASE 4 SKIPPED: Max refinements ({self.max_refinements}) reached")

        # PHASE 5: Finalize Results
        total_time = (time.time() - start_time) * 1000

        logger.info(f"🚀 ENHANCED AGENT COMPLETE: Total time {total_time:.0f}ms "
                   f"(planning: {planning_time:.0f}ms, execution: {execution_time:.0f}ms, "
                   f"evaluation: {evaluation_time:.0f}ms, refinement: {refinement_time:.0f}ms)")

        # Build enhanced result
        enhanced_result = {
            **execution_result,  # Include all original results
            "enhanced_agent_metadata": {
                "planning": {
                    "complexity": planning_result.estimated_complexity if planning_result else "unknown",
                    "reasoning": planning_result.reasoning if planning_result else "Fallback planning used",
                    "steps_planned": len(planning_result.execution_plan) if planning_result else 0,
                    "time_ms": planning_time
                },
                "evaluation": {
                    "quality_score": evaluation.quality_score if evaluation else None,
                    "relevance_score": evaluation.relevance_score if evaluation else None,
                    "coverage_score": evaluation.coverage_score if evaluation else None,
                    "confidence_score": evaluation.confidence_score if evaluation else None,
                    "reasoning": evaluation.reasoning if evaluation else "Evaluation unavailable",
                    "suggestions": evaluation.suggestions if evaluation else [],
                    "time_ms": evaluation_time
                } if evaluation else None,
                "refinement": {
                    "executed": refinement_executed,
                    "count": self.refinement_count,
                    "time_ms": refinement_time
                },
                "performance": {
                    "total_time_ms": total_time,
                    "planning_time_ms": planning_time,
                    "execution_time_ms": execution_time,
                    "evaluation_time_ms": evaluation_time,
                    "refinement_time_ms": refinement_time
                },
                "execution_log": self.execution_log
            }
        }

        return enhanced_result

    def _merge_results(self, original: Dict[str, Any], refinement: Dict[str, Any]) -> Dict[str, Any]:
        """Merge refinement results with original results.

        Args:
            original: Original execution results
            refinement: Refinement execution results

        Returns:
            Merged results with deduplication
        """
        merged = original.copy()

        # Merge documents
        original_docs = {doc.get('document_id'): doc for doc in original.get('documents', [])}
        refinement_docs = {doc.get('document_id'): doc for doc in refinement.get('documents', [])}

        # Add new documents from refinement
        for doc_id, doc in refinement_docs.items():
            if doc_id not in original_docs:
                original_docs[doc_id] = doc

        merged['documents'] = list(original_docs.values())

        # Merge chunks
        original_chunks = {chunk.get('chunk_id'): chunk for chunk in original.get('document_chunks', [])}
        refinement_chunks = {chunk.get('chunk_id'): chunk for chunk in refinement.get('document_chunks', [])}

        # Add new chunks from refinement
        for chunk_id, chunk in refinement_chunks.items():
            if chunk_id not in original_chunks:
                original_chunks[chunk_id] = chunk

        merged['document_chunks'] = list(original_chunks.values())

        logger.info(f"🔄 MERGE: Combined results - {len(merged['documents'])} total documents, "
                   f"{len(merged['document_chunks'])} total chunks")

        return merged

    def _log_step(self, step_data: Dict[str, Any]):
        """Log execution step for tracking and analysis.

        Args:
            step_data: Step metadata to log
        """
        step_data['timestamp'] = time.time()
        self.execution_log.append(step_data)


def create_enhanced_agent(base_agent, llm_client) -> EnhancedVectorSearchAgent:
    """Factory function to create an enhanced agent.

    Args:
        base_agent: Existing VectorSearchAgent instance
        llm_client: LLM client

    Returns:
        EnhancedVectorSearchAgent instance
    """
    return EnhancedVectorSearchAgent(base_agent, llm_client)
