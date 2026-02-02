"""Structured logging utilities for enhanced agent execution.

Provides detailed, structured logging for agent reasoning steps
to improve observability and debugging.
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentExecutionLogger:
    """Structured logger for agent execution with detailed reasoning traces."""

    def __init__(self, query: str, session_id: str = None):
        """Initialize agent execution logger.

        Args:
            query: The user query being processed
            session_id: Optional session identifier
        """
        self.query = query
        self.session_id = session_id or self._generate_session_id()
        self.execution_trace = []
        self.start_time = datetime.now()

    def log_planning_start(self, reason: str, context: Dict[str, Any]):
        """Log the start of planning phase.

        Args:
            reason: Reason for agent mode selection
            context: Execution context
        """
        self._add_trace({
            "phase": "planning",
            "event": "start",
            "reason": reason,
            "context_summary": self._summarize_context(context),
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🧠 [SESSION {self.session_id}] PLANNING START - Query: '{self.query}' | Reason: {reason}")

    def log_planning_analysis(self, analysis: Dict[str, Any]):
        """Log query analysis results.

        Args:
            analysis: Query analysis results
        """
        self._add_trace({
            "phase": "planning",
            "event": "analysis",
            "complexity": analysis.get('complexity', 'unknown'),
            "intent": analysis.get('intent', 'unknown'),
            "key_concepts": analysis.get('key_concepts', []),
            "risk_factors": analysis.get('risk_factors', []),
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🧠 [SESSION {self.session_id}] ANALYSIS - "
                   f"Complexity: {analysis.get('complexity', 'unknown')} | "
                   f"Intent: {analysis.get('intent', 'unknown')} | "
                   f"Concepts: {len(analysis.get('key_concepts', []))}")

    def log_planning_complete(self, plan: List[Dict[str, Any]], reasoning: str):
        """Log planning completion.

        Args:
            plan: Execution plan created
            reasoning: Planning reasoning
        """
        self._add_trace({
            "phase": "planning",
            "event": "complete",
            "steps_count": len(plan),
            "step_names": [step.get('step_name', 'unknown') for step in plan],
            "reasoning": reasoning[:500],  # Truncate for logging
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🧠 [SESSION {self.session_id}] PLANNING COMPLETE - "
                   f"{len(plan)} steps | Reasoning: {reasoning[:100]}...")

    def log_execution_start(self, plan: List[Dict[str, Any]]):
        """Log the start of execution phase.

        Args:
            plan: Execution plan to execute
        """
        self._add_trace({
            "phase": "execution",
            "event": "start",
            "steps_to_execute": len(plan),
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🔍 [SESSION {self.session_id}] EXECUTION START - {len(plan)} steps to execute")

    def log_step_start(self, step_index: int, step: Dict[str, Any]):
        """Log the start of a specific execution step.

        Args:
            step_index: Index of the step
            step: Step details
        """
        self._add_trace({
            "phase": "execution",
            "event": "step_start",
            "step_index": step_index,
            "step_name": step.get('step_name', 'unknown'),
            "tool": step.get('tool', 'unknown'),
            "reasoning": step.get('reasoning', ''),
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🔍 [SESSION {self.session_id}] STEP {step_index + 1} START - "
                   f"{step.get('step_name', 'unknown')} ({step.get('tool', 'unknown')}) | "
                   f"Reasoning: {step.get('reasoning', 'N/A')[:80]}...")

    def log_step_complete(self, step_index: int, step_name: str, result: Dict[str, Any],
                         duration_ms: float):
        """Log the completion of a specific execution step.

        Args:
            step_index: Index of the step
            step_name: Name of the step
            result: Step execution result
            duration_ms: Execution duration in milliseconds
        """
        success = result.get('success', False)

        self._add_trace({
            "phase": "execution",
            "event": "step_complete",
            "step_index": step_index,
            "step_name": step_name,
            "success": success,
            "duration_ms": duration_ms,
            "result_summary": self._summarize_result(result),
            "timestamp": datetime.now().isoformat()
        })

        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"🔍 [SESSION {self.session_id}] STEP {step_index + 1} COMPLETE - "
                   f"{step_name} | {status} | {duration_ms:.0f}ms")

    def log_execution_complete(self, results: Dict[str, Any], duration_ms: float):
        """Log the completion of execution phase.

        Args:
            results: Execution results
            duration_ms: Total execution duration
        """
        self._add_trace({
            "phase": "execution",
            "event": "complete",
            "documents_found": len(results.get('documents', [])),
            "chunks_found": len(results.get('document_chunks', [])),
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🔍 [SESSION {self.session_id}] EXECUTION COMPLETE - "
                   f"Found {len(results.get('documents', []))} docs, "
                   f"{len(results.get('document_chunks', []))} chunks | {duration_ms:.0f}ms")

    def log_evaluation_start(self):
        """Log the start of evaluation phase."""
        self._add_trace({
            "phase": "evaluation",
            "event": "start",
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"📊 [SESSION {self.session_id}] EVALUATION START")

    def log_evaluation_complete(self, evaluation: Dict[str, Any], duration_ms: float):
        """Log the completion of evaluation phase.

        Args:
            evaluation: Evaluation results
            duration_ms: Evaluation duration
        """
        quality_score = evaluation.get('quality_score', 0.0)
        needs_refinement = evaluation.get('needs_refinement', False)

        self._add_trace({
            "phase": "evaluation",
            "event": "complete",
            "quality_score": quality_score,
            "relevance_score": evaluation.get('relevance_score', 0.0),
            "coverage_score": evaluation.get('coverage_score', 0.0),
            "confidence_score": evaluation.get('confidence_score', 0.0),
            "needs_refinement": needs_refinement,
            "reasoning": evaluation.get('reasoning', '')[:500],
            "suggestions": evaluation.get('suggestions', []),
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        })

        status = "🔄 NEEDS REFINEMENT" if needs_refinement else "✅ ACCEPTABLE"
        logger.info(f"📊 [SESSION {self.session_id}] EVALUATION COMPLETE - "
                   f"Quality: {quality_score:.2f} | {status} | {duration_ms:.0f}ms")

    def log_refinement_start(self, reason: str, plan_size: int):
        """Log the start of refinement phase.

        Args:
            reason: Reason for refinement
            plan_size: Number of refinement steps
        """
        self._add_trace({
            "phase": "refinement",
            "event": "start",
            "reason": reason,
            "steps_count": plan_size,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🔄 [SESSION {self.session_id}] REFINEMENT START - "
                   f"{plan_size} steps | Reason: {reason}")

    def log_refinement_complete(self, improvement: Dict[str, Any], duration_ms: float):
        """Log the completion of refinement phase.

        Args:
            improvement: Improvement metrics
            duration_ms: Refinement duration
        """
        self._add_trace({
            "phase": "refinement",
            "event": "complete",
            "improvement": improvement,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"🔄 [SESSION {self.session_id}] REFINEMENT COMPLETE - "
                   f"Quality improved: {improvement.get('quality_improved', False)} | "
                   f"{duration_ms:.0f}ms")

    def log_error(self, phase: str, error: Exception):
        """Log an error during execution.

        Args:
            phase: Phase where error occurred
            error: Exception that occurred
        """
        self._add_trace({
            "phase": phase,
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        })

        logger.error(f"❌ [SESSION {self.session_id}] ERROR in {phase.upper()} - "
                    f"{type(error).__name__}: {str(error)}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire execution.

        Returns:
            Dictionary with execution summary
        """
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() * 1000

        phases = {}
        for trace in self.execution_trace:
            phase = trace['phase']
            if phase not in phases:
                phases[phase] = {
                    'events': 0,
                    'errors': 0,
                    'duration_ms': 0
                }
            phases[phase]['events'] += 1
            if trace.get('event') == 'error':
                phases[phase]['errors'] += 1
            if 'duration_ms' in trace:
                phases[phase]['duration_ms'] += trace['duration_ms']

        return {
            "session_id": self.session_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_ms": total_duration,
            "phases": phases,
            "trace_length": len(self.execution_trace),
            "errors_count": sum(p['errors'] for p in phases.values())
        }

    def export_trace(self, filepath: str = None) -> str:
        """Export execution trace to JSON.

        Args:
            filepath: Optional path to save trace file

        Returns:
            JSON string of execution trace
        """
        trace_export = {
            "session_id": self.session_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "summary": self.get_execution_summary(),
            "trace": self.execution_trace
        }

        trace_json = json.dumps(trace_export, indent=2)

        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(trace_json)
                logger.info(f"📝 [SESSION {self.session_id}] Trace exported to {filepath}")
            except Exception as e:
                logger.error(f"❌ [SESSION {self.session_id}] Failed to export trace: {e}")

        return trace_json

    def _add_trace(self, trace_entry: Dict[str, Any]):
        """Add an entry to the execution trace.

        Args:
            trace_entry: Trace entry to add
        """
        self.execution_trace.append(trace_entry)

    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize context for logging.

        Args:
            context: Full execution context

        Returns:
            Summarized context
        """
        return {
            "has_projects": bool(context.get('discovered_project_ids')),
            "has_document_types": bool(context.get('discovered_document_type_ids')),
            "user_provided_filters": bool(
                context.get('user_project_ids') or context.get('user_document_type_ids')
            )
        }

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize result for logging.

        Args:
            result: Full result

        Returns:
            Summarized result
        """
        if not isinstance(result, dict):
            return {"type": type(result).__name__}

        summary = {
            "success": result.get('success', False)
        }

        # Add specific fields based on result type
        if 'documents' in result:
            summary['documents_count'] = len(result['documents'])
        if 'document_chunks' in result:
            summary['chunks_count'] = len(result['document_chunks'])
        if 'error' in result:
            summary['error'] = result['error']
        if 'total_results' in result:
            summary['total_results'] = result['total_results']

        return summary

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID.

        Returns:
            Session ID string
        """
        import uuid
        return str(uuid.uuid4())[:8]
