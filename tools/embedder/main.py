import argparse
import logging
from datetime import datetime
from src.services.logger import load_completed_files, load_incomplete_files, load_skipped_files
from src.services.repair_service import get_repair_candidates_for_processing, cleanup_document_data, cleanup_project_data, cleanup_document_content_for_retry, print_repair_analysis
from src.models.pgvector.vector_db_utils import init_vec_db
from src.config.settings import get_settings
from src.utils.error_suppression import suppress_process_pool_errors
import tempfile
import uuid
import shutil
import os

"""
EPIC.search Embedder - Main Entry Point

This module serves as the main entry point for the EPIC.search Embedder application,
which processes PDF documents from projects and converts them into vector embeddings
for semantic search capabilities.

The application can process a single project (when a project_id is provided) or
all available projects. It tracks processing status to avoid re-processing documents
and handles parallel processing for efficiency.
"""

from src.services.api_utils import (
    get_files_count_for_project,
    get_project_by_id,
    get_projects,
    get_files_for_project,
    get_projects_count,
)
from src.services.processor import process_files
from src.services.data_formatter import format_metadata
from src.services.project_utils import upsert_project
from src.services.ocr.ocr_factory import initialize_ocr
from src.utils.progress_tracker import progress_tracker

# Initialize settings at module level
settings = get_settings()

def setup_logging():
    """
    Configure the logging system for the application.
    
    Sets up basic logging configuration with INFO level and a standard format.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def is_document_already_processed(doc_id, completed_docs, incomplete_docs, skipped_docs):
    """
    Check if a document has been previously processed (successfully, unsuccessfully, or skipped).
    
    Args:
        doc_id: The document ID to check
        completed_docs: List of successfully processed documents
        incomplete_docs: List of unsuccessfully processed documents
        skipped_docs: List of skipped documents
        
    Returns:
        tuple: (is_processed, status_message) where is_processed is a boolean and 
               status_message is a string or None if not processed
    """
    for completed in completed_docs:
        if completed["document_id"] == doc_id:
            return True, "success"
            
    for incomplete in incomplete_docs:
        if incomplete["document_id"] == doc_id:
            return True, "failed"
    
    for skipped in skipped_docs:
        if skipped["document_id"] == doc_id:
            return True, "skipped"
            
    return False, None

def get_embedder_temp_dir():
    temp_root = tempfile.gettempdir()
    temp_guid = str(uuid.uuid4())
    temp_dir = os.path.join(temp_root, f"epic_embedder_{temp_guid}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def check_time_limit(start_time, time_limit_seconds):
    """
    Check if the time limit has been reached for timed mode.
    
    Args:
        start_time (datetime): The start time of processing
        time_limit_seconds (int): Time limit in seconds
        
    Returns:
        tuple: (time_limit_reached, elapsed_minutes, remaining_minutes)
    """
    if time_limit_seconds is None:
        return False, 0, 0
    
    elapsed_time = datetime.now() - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    elapsed_minutes = elapsed_seconds / 60
    remaining_seconds = max(0, time_limit_seconds - elapsed_seconds)
    remaining_minutes = remaining_seconds / 60
    
    return elapsed_seconds >= time_limit_seconds, elapsed_minutes, remaining_minutes

def process_projects(project_ids=None, shallow_mode=False, shallow_limit=None, skip_hnsw_indexes=False, retry_failed_only=False, retry_skipped_only=False, repair_mode=False, timed_mode=False, time_limit_minutes=None):
    """
    Process documents for one or more specific projects, or all projects.
    
    This function:
    1. Initializes the database connections
    2. Fetches project information from the API
    3. For each project, retrieves its documents
    4. Filters out already processed documents (or includes only failed/skipped ones in retry mode)
    5. Processes documents using continuous queue with dynamic worker allocation
    
    Args:
        project_ids (list or str, optional): Process specific project(s). Can be a single ID string or list of IDs. If None, all projects are processed.
        shallow_mode (bool, optional): If True, only process up to shallow_limit successful documents per project.
        shallow_limit (int, optional): The maximum number of successful documents to process per project in shallow mode.
        skip_hnsw_indexes (bool, optional): Skip creation of HNSW vector indexes for faster startup.
        retry_failed_only (bool, optional): If True, only process documents that previously failed processing.
        retry_skipped_only (bool, optional): If True, only process documents that were previously skipped.
        repair_mode (bool, optional): If True, identify and repair documents in inconsistent states.
        timed_mode (bool, optional): If True, run in timed mode with time limit.
        time_limit_minutes (int, optional): Time limit in minutes for timed mode processing.
        
    Returns:
        dict: A dictionary containing the processing results, including:
            - message: A status message
            - results: A list of project processing results, each with:
                - project_name: Name of the processed project
                - duration_seconds: Time taken to process the project
    """
    
    # Print performance configuration
    files_concurrency = settings.multi_processing_settings.files_concurrency_size
    keyword_workers = settings.multi_processing_settings.keyword_extraction_workers
    chunk_batch_size = settings.multi_processing_settings.chunk_insert_batch_size
    
    print(f"[PERF] Document workers: {files_concurrency}")
    print(f"[PERF] Keyword threads per document: {keyword_workers}")
    print(f"[PERF] Database batch size: {chunk_batch_size}")
    print(f"[PERF] Total potential keyword threads: {files_concurrency * keyword_workers}")    
    
    # Show processing mode
    if retry_failed_only and retry_skipped_only:
        print(f"[MODE] RETRY FAILED & SKIPPED MODE: Processing documents that previously failed OR were skipped")
    elif retry_failed_only:
        print(f"[MODE] RETRY FAILED MODE: Only processing documents that previously failed")
    elif retry_skipped_only:
        print(f"[MODE] RETRY SKIPPED MODE: Only processing documents that were previously skipped")
    elif repair_mode:
        print(f"[MODE] REPAIR MODE: Identifying and fixing documents in inconsistent states")
    else:
        print(f"[MODE] NORMAL MODE: Processing new documents (skipping successful ones)")
    
    # Initialize timing for timed mode
    start_time = datetime.now()
    time_limit_reached = False
    
    if timed_mode:
        print(f"[TIMED MODE] Running for {time_limit_minutes} minutes. Will gracefully stop after time limit.")
        print(f"[TIMED MODE] Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time_limit_seconds = time_limit_minutes * 60
    else:
        time_limit_seconds = None
    
    # Pass skip_hnsw_indexes from main
    init_vec_db(skip_hnsw=skip_hnsw_indexes)

    if project_ids:
        # Process specific project(s)
        # Handle both single string and list of strings
        if isinstance(project_ids, str):
            project_ids = [project_ids]
        
        projects = []
        for project_id in project_ids:
            project_data = get_project_by_id(project_id)
            if project_data:
                projects.extend(project_data)
            else:
                print(f"Warning: Project ID '{project_id}' not found")
    else:
        # Fetch and process all projects
        projects_count = get_projects_count()
        page_size = settings.api_pagination_settings.project_page_size
        total_pages = (
            projects_count + page_size - 1
        ) // page_size  # Calculate total pages

        projects = []
        for page_number in range(total_pages):
            projects.extend(get_projects(page_number, page_size))

    if not projects:
        return {"message": "No projects returned by API."}

    # Calculate total documents for progress tracking
    total_documents = 0
    for project in projects:
        try:
            project_id = project["_id"]
            files_count = get_files_count_for_project(project_id)
            total_documents += files_count
        except Exception as e:
            print(f"Warning: Could not get file count for project {project.get('name', 'unknown')}: {e}")
    
    # Start progress tracking
    print(f"TIMED MODE: {time_limit_minutes} minutes limit" if timed_mode else "")
    progress_tracker.start(len(projects), total_documents)

    results = []
    embedder_temp_dir = get_embedder_temp_dir()
    
    # Determine processing mode: dynamic queuing or sequential
    # Use dynamic queuing for all cases - even shallow mode can benefit from continuous queue
    # within the per-project limits
    use_dynamic_processing = True  # Always use dynamic processing for maximum efficiency
    
    print(f"[DEBUG] Processing mode determination:")
    print(f"[DEBUG]   - Number of projects: {len(projects)}")
    print(f"[DEBUG]   - Shallow mode: {shallow_mode}")
    print(f"[DEBUG]   - Use dynamic processing: {use_dynamic_processing}")
    
    if use_dynamic_processing:
        mode_type = "NORMAL"
        if retry_failed_only and retry_skipped_only:
            mode_type = "RETRY FAILED & SKIPPED"
        elif retry_failed_only:
            mode_type = "RETRY FAILED"
        elif retry_skipped_only:
            mode_type = "RETRY SKIPPED"
        elif repair_mode:
            mode_type = "REPAIR"
            
        print(f"\n=== DYNAMIC {mode_type} PROCESSING MODE ===")
        if len(projects) > 1:
            print(f"Processing {len(projects)} projects with unified worker pool to maximize throughput")
            print(f"All {settings.multi_processing_settings.files_concurrency_size} workers will stay busy across projects")
            print(f"CROSS-PROJECT QUEUE: Documents from different projects will be processed in continuous queue for optimal worker utilization")
        elif shallow_mode:
            print(f"Processing {len(projects)} projects with shallow limits using continuous queue")
            print(f"All {settings.multi_processing_settings.files_concurrency_size} workers will stay busy within per-project limits")
            print(f"SHALLOW + DYNAMIC: Up to {shallow_limit} documents per project, but processed with full worker utilization")
        else:
            print(f"Processing single project with dynamic worker queuing to maximize throughput")
            print(f"All {settings.multi_processing_settings.files_concurrency_size} workers will stay busy with continuous document queuing")
            print(f"DYNAMIC QUEUING: Workers immediately get next document when they finish current one")
        results = process_projects_in_parallel(projects, embedder_temp_dir, start_time, timed_mode, time_limit_seconds, retry_failed_only, retry_skipped_only, repair_mode, shallow_mode, shallow_limit)
    else:
        # This branch should never be reached now since use_dynamic_processing is always True
        print(f"\n=== FALLBACK SEQUENTIAL PROCESSING ===")
        print(f"This should not happen - please report as a bug")
        
        # Original sequential project processing loop
        for project in projects:
            # Check time limit before starting each project
            if timed_mode:
                time_limit_reached, elapsed_minutes, remaining_minutes = check_time_limit(start_time, time_limit_seconds)
                if time_limit_reached:
                    print(f"\n[TIMED MODE] Time limit of {time_limit_minutes} minutes reached after {elapsed_minutes:.1f} minutes.")
                    print(f"[TIMED MODE] Gracefully stopping. Completed {len(results)} project(s).")
                    break
                else:
                    print(f"[TIMED MODE] Elapsed: {elapsed_minutes:.1f}min, Remaining: {remaining_minutes:.1f}min")
            
            project_id = project["_id"]
            project_name = project["name"]

            # Update progress tracker with current project
            progress_tracker.update_current_project(project_name)

            # Ensure project record exists before processing documents
            upsert_project(project_id, project_name, project)

            print(
                f"\n=== Retrieving documents for project: {project_name} ({project_id}) ==="
            )

            if shallow_mode:
                print(f"[SHALLOW MODE] Will process up to {shallow_limit} successful documents for this project.")

            project_start = datetime.now()

            files_count = get_files_count_for_project(project_id)
            page_size = settings.api_pagination_settings.documents_page_size
            file_total_pages = (
                files_count + page_size - 1
            ) // page_size  # Calculate total pages for files

            already_completed = load_completed_files(project_id)
            already_incomplete = load_incomplete_files(project_id)
            already_skipped = load_skipped_files(project_id)

            # Handle repair mode - analyze and process inconsistent documents
            if repair_mode:
                print(f"\n REPAIR MODE: Analyzing {project_name} for inconsistent document states...")
                print(f"REPAIR MODE: Skipping normal document processing flow")
                
                # Get repair candidates for this project
                repair_candidates = get_repair_candidates_for_processing(project_id)
                
                if not repair_candidates:
                    print(f"No documents need repair for {project_name} - database may be empty or all documents are consistent")
                    continue
                
                print(f"Found {len(repair_candidates)} documents that need repair for {project_name}")
                
                # Process repair candidates instead of normal document flow
                s3_file_keys = []
                metadata_list = []
                docs_to_process = []
                api_docs_list = []
                
                # Get the actual document data for repair candidates
                for candidate in repair_candidates:
                    doc_id = candidate['document_id']
                    
                    # Clean up existing inconsistent data first
                    print(f"Cleaning up inconsistent data for {candidate['document_name'][:50]}...")
                    cleanup_summary = cleanup_document_data(doc_id, project_id)
                    print(f"Deleted: {cleanup_summary['chunks_deleted']} chunks, {cleanup_summary['document_records_deleted']} docs, {cleanup_summary['processing_logs_deleted']} logs")
                    
                    # Find the document in the API to reprocess it
                    # We need to search through all file pages to find this specific document
                    doc_found = False
                    for search_page in range(file_total_pages):
                        search_files = get_files_for_project(project_id, search_page, page_size)
                        for doc in search_files:
                            if doc["_id"] == doc_id:
                                doc_found = True
                                s3_key = doc.get("internalURL")
                                if s3_key:
                                    doc_meta = format_metadata(project, doc)
                                    s3_file_keys.append(s3_key)
                                    metadata_list.append(doc_meta)
                                    docs_to_process.append(doc)
                                    api_docs_list.append(doc)
                                    print(f" Queued for reprocessing: {candidate['document_name'][:50]}")
                                break
                        if doc_found:
                            break
                    
                    if not doc_found:
                        print(f"Warning: Could not find document {doc_id} in API - may have been deleted")
                
                # Process the repair candidates
                if s3_file_keys:
                    files_concurrency_size = settings.multi_processing_settings.files_concurrency_size
                    print(f"Reprocessing {len(s3_file_keys)} repaired documents for {project_name} with {files_concurrency_size} workers...")
                    
                    process_files(
                        project_id,
                        s3_file_keys,
                        metadata_list,
                        api_docs_list,
                        files_concurrency_size,
                        embedder_temp_dir,
                        is_retry=False,
                    )
                    
                    print(f" Repair completed for {project_name}")
                else:
                    print(f" No repairable documents found in API for {project_name}")
                
                # Skip to next project in repair mode
                continue

            # For shallow mode, count how many have been processed successfully
            shallow_success_count = len(already_completed) if shallow_mode else 0

            for file_page_number in range(file_total_pages):
                # Check time limit before processing each page of files
                if timed_mode:
                    time_limit_reached, elapsed_minutes, remaining_minutes = check_time_limit(start_time, time_limit_seconds)
                    if time_limit_reached:
                        print(f"[TIMED MODE] Time limit reached during {project_name}. Stopping file processing for this project.")
                        break
                
                if shallow_mode and shallow_success_count >= shallow_limit:
                    print(f"[SHALLOW MODE] {shallow_success_count} documents already processed for {project_name}, skipping rest.")
                    break
                files_data = get_files_for_project(project_id, file_page_number, page_size)

                if not files_data:
                    print(f"No files found for project {project_id}")
                    continue

                s3_file_keys = []
                metadata_list = []
                docs_to_process = []
                api_docs_list = []  # Store API document objects

                for doc in files_data:
                    # Check time limit before processing each document
                    if timed_mode:
                        time_limit_reached, elapsed_minutes, remaining_minutes = check_time_limit(start_time, time_limit_seconds)
                        if time_limit_reached:
                            print(f"[TIMED MODE] Time limit reached during document processing for {project_name}. Stopping at {len(s3_file_keys)} documents queued.")
                            break
                    
                    doc_id = doc["_id"]
                    doc_name = doc.get('name', doc_id)
                    is_processed, status = is_document_already_processed(
                        doc_id, already_completed, already_incomplete, already_skipped
                    )
                    
                    # Handle retry modes
                    if retry_failed_only and retry_skipped_only:
                        # Combined mode: process both failed and skipped documents
                        if not is_processed or (status != "failed" and status != "skipped"):
                            continue
                        
                        status_type = "failed" if status == "failed" else "skipped"
                        print(f"Retrying {status_type} document: {doc_name}")
                        
                    elif retry_failed_only:
                        # Only process files that previously failed
                        if not is_processed or status != "failed":
                            continue
                        print(f"Retrying failed document: {doc_name}")
                        
                    elif retry_skipped_only:
                        # Only process files that were previously skipped
                        if not is_processed or status != "skipped":
                            continue
                        print(f"Retrying skipped document: {doc_name}")
                        
                    else:
                        # Normal mode: skip already processed files
                        if is_processed:
                            print(f"Skipping already processed ({status}) document: {doc_name}")
                            continue
                            
                    s3_key = doc.get("internalURL")
                    if not s3_key:
                        continue
                    doc_meta = format_metadata(project, doc)
                    s3_file_keys.append(s3_key)
                    metadata_list.append(doc_meta)
                    docs_to_process.append(doc)
                    api_docs_list.append(doc)  # Store the API document object
                    if shallow_mode and (shallow_success_count + len(s3_file_keys)) >= shallow_limit:
                        # Only add up to the shallow limit
                        break

                if s3_file_keys:
                    files_concurrency_size = settings.multi_processing_settings.files_concurrency_size
                    if retry_failed_only and retry_skipped_only:
                        print(
                            f"Found {len(s3_file_keys)} failed/skipped file(s) to retry for {project_name}. Processing with {files_concurrency_size} workers..."
                        )
                    elif retry_failed_only:
                        print(
                            f"Found {len(s3_file_keys)} failed file(s) to retry for {project_name}. Processing with {files_concurrency_size} workers..."
                        )
                    elif retry_skipped_only:
                        print(
                            f"Found {len(s3_file_keys)} skipped file(s) to retry for {project_name}. Processing with {files_concurrency_size} workers..."
                        )
                    else:
                        print(
                            f"Found {len(s3_file_keys)} new file(s) for {project_name}. Processing with {files_concurrency_size} workers..."
                        )
                    
                    # This fallback code is obsolete - all processing now uses continuous queue
                    # The following code should never execute as process_projects_in_parallel handles all modes
                    print(f"[WARNING] Fallback processing detected - this should not happen. Using continuous queue anyway.")
                    
                    # Convert to dynamic processing format and delegate to the unified processor
                    from collections import namedtuple
                    DocumentTask = namedtuple('DocumentTask', ['project_id', 'project_name', 's3_key', 'metadata', 'api_doc'])
                    document_queue = []
                    
                    for i, s3_key in enumerate(s3_file_keys):
                        task = DocumentTask(
                            project_id=project_id,
                            project_name=project_name,
                            s3_key=s3_key,
                            metadata=metadata_list[i],
                            api_doc=api_docs_list[i]
                        )
                        document_queue.append(task)
                    
                    # Use the unified continuous queue processor
                    from src.services.loader import load_data
                    processing_result = process_mixed_project_files(
                        document_queue,
                        [task.s3_key for task in document_queue],
                        [task.metadata for task in document_queue],
                        [task.api_doc for task in document_queue],
                        batch_size=files_concurrency_size,
                        temp_dir=embedder_temp_dir,
                        is_retry=False,
                        timed_mode=timed_mode,
                        time_limit_seconds=time_limit_seconds,
                        start_time=start_time,
                    )
                        
                    # Update shallow count for normal mode
                    if not timed_mode and shallow_mode:
                        shallow_success_count += len(s3_file_keys)
                        if shallow_success_count >= shallow_limit:
                            print(f"[SHALLOW MODE] Reached {shallow_success_count} processed documents for {project_name} (limit: {shallow_limit}).")
                            break
                else:
                    if retry_failed_only and retry_skipped_only:
                        print(f"No failed or skipped files found to retry for {project_name}")
                    elif retry_failed_only:
                        print(f"No failed files found to retry for {project_name}")
                    elif retry_skipped_only:
                        print(f"No skipped files found to retry for {project_name}")
                    else:
                        print(f"No new files to process for {project_name}")

            project_end = datetime.now()
            duration = project_end - project_start
            duration_in_s = duration.total_seconds()
            print(
                f"Project processing completed for {project_name} in {duration_in_s} seconds"
            )
            
            # Mark project as completed in progress tracker
            progress_tracker.finish_project()
            
            results.append(
                {"project_name": project_name, "duration_seconds": duration_in_s}
            )

    # After all processing, clean up the temp folder
    print(f"Cleaning up embedder temp directory: {embedder_temp_dir}")
    try:
        shutil.rmtree(embedder_temp_dir)
        print("Temp directory cleaned up.")
    except Exception as cleanup_err:
        print(f"[WARN] Could not delete temp directory {embedder_temp_dir}: {cleanup_err}")

    # Summary message
    total_elapsed = datetime.now() - start_time
    total_elapsed_minutes = total_elapsed.total_seconds() / 60
    
    if timed_mode:
        if time_limit_reached:
            print(f"TIMED MODE COMPLETED: Stopped after {total_elapsed_minutes:.1f} minutes (limit: {time_limit_minutes} minutes)")
        else:
            print(f"TIMED MODE COMPLETED: Finished all work in {total_elapsed_minutes:.1f} minutes (limit: {time_limit_minutes} minutes)")
        
        if retry_failed_only and retry_skipped_only:
            print(f"Processed failed and skipped documents for {len(results)} project(s)")
        elif retry_failed_only:
            print(f"Processed failed documents for {len(results)} project(s)")
        elif retry_skipped_only:
            print(f"Processed skipped documents for {len(results)} project(s)")
        else:
            print(f"Processed new documents for {len(results)} project(s)")
    else:
        if retry_failed_only and retry_skipped_only:
            print(f"RETRY COMPLETED: Finished retrying failed and skipped documents for {len(results)} project(s)")
        elif retry_failed_only:
            print(f"RETRY COMPLETED: Finished retrying failed documents for {len(results)} project(s)")
        elif retry_skipped_only:
            print(f"RETRY COMPLETED: Finished retrying skipped documents for {len(results)} project(s)")
        else:
            print(f"PROCESSING COMPLETED: Finished processing new documents for {len(results)} project(s)")

    # Stop progress tracking
    reason = "Completed"
    if timed_mode and time_limit_reached:
        reason = f"Time limit reached ({time_limit_minutes} minutes)"
    elif retry_failed_only and retry_skipped_only:
        reason = "Retry failed and skipped mode completed"
    elif retry_failed_only:
        reason = "Retry failed mode completed"
    elif retry_skipped_only:
        reason = "Retry skipped mode completed"
    
    progress_tracker.stop(reason)

    return {"message": "Processing completed", "results": results}


def process_mixed_project_files(document_tasks, file_keys, metadata_list, api_docs_list, batch_size=4, temp_dir=None, is_retry=False, timed_mode=False, time_limit_seconds=None, start_time=None):
    """
    Process files from multiple projects concurrently using a unified worker pool with dynamic queuing.
    Workers continuously pull new documents as they finish, maximizing CPU utilization.
    
    Args:
        document_tasks (list): List of DocumentTask namedtuples containing project info for each document
        file_keys (list): List of file keys or paths to be processed
        metadata_list (list): List of metadata dictionaries corresponding to each file  
        api_docs_list (list): List of API document objects corresponding to each file
        batch_size (int, optional): Number of worker processes to run in parallel. Defaults to 4.
        temp_dir (str, optional): Temporary directory for processing files. Passed to load_data.
        is_retry (bool, optional): If True, cleanup existing document content before processing.
        timed_mode (bool, optional): If True, check time limits and stop gracefully when reached.
        time_limit_seconds (int, optional): Time limit in seconds for timed mode.
        start_time (datetime, optional): Start time for timed mode calculations.
        
    Returns:
        dict: Processing results including time_limit_reached status
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from src.services.logger import log_processing_result
    from src.services.loader import load_data
    from datetime import datetime
    import os
    
    if len(file_keys) != len(metadata_list) or len(file_keys) != len(api_docs_list) or len(file_keys) != len(document_tasks):
        raise ValueError("document_tasks, file_keys, metadata_list, and api_docs_list must have the same length.")

    if not file_keys:
        print("No files to process.")
        return {"time_limit_reached": False, "documents_processed": 0}

    print(f"Starting ProcessPoolExecutor with max_workers={batch_size}")
    print(f"DYNAMIC QUEUING: Workers will continuously pull new documents as they finish")
    
    if timed_mode and time_limit_seconds and start_time:
        print(f"TIMED MODE: Will stop gracefully when {time_limit_seconds/60:.1f} minute limit is reached")
    
    time_limit_reached = False
    documents_processed = 0
    
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        future_to_task = {}
        document_index = 0
        total_documents = len(document_tasks)
        completed_count = 0
        
        # Submit initial worker pool (up to worker_count workers)
        while document_index < min(batch_size, total_documents):
            # Check time limit before submitting initial work
            if timed_mode and time_limit_seconds and start_time:
                elapsed_time = datetime.now() - start_time
                if elapsed_time.total_seconds() >= time_limit_seconds:
                    print(f"[TIMED MODE] Time limit reached before processing could begin. Stopping gracefully.")
                    time_limit_reached = True
                    break
            
            task = document_tasks[document_index]
            file_key = file_keys[document_index]
            base_meta = metadata_list[document_index]
            api_doc = api_docs_list[document_index]
            
            worker_id = document_index % batch_size + 1
            doc_id = base_meta.get("document_id") or os.path.basename(file_key)
            doc_name = api_doc.get('name', doc_id)
            
            # File size handling
            file_size_raw = api_doc.get('internalSize', '0')
            try:
                file_size_bytes = int(file_size_raw) if file_size_raw else 0
            except (ValueError, TypeError):
                file_size_bytes = 0
            
            size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else None
            estimated_pages = max(1, int(file_size_bytes / 50000)) if file_size_bytes > 0 else None
            
            # Submit work and track it
            future = executor.submit(load_data, file_key, base_meta, temp_dir, api_doc, is_retry)
            future_to_task[future] = (task, file_key, base_meta, api_doc, worker_id, doc_name, estimated_pages, size_mb, document_index)
            
            # Update progress tracker
            progress_tracker.update_current_project(task.project_name)
            progress_tracker.start_document_processing(worker_id, doc_name, estimated_pages, size_mb)
            
            document_index += 1
        
        if not time_limit_reached:
            print(f"Submitted initial {len(future_to_task)} documents to workers. Queue contains {total_documents - document_index} more documents.")
        
        # Process completed tasks and dynamically submit new work
        for future in as_completed(future_to_task):
            task, file_key, base_meta, api_doc, worker_id, doc_name, estimated_pages, size_mb, doc_index = future_to_task[future]
            doc_id = base_meta.get("document_id") or os.path.basename(file_key)
            project_id = task.project_id
            
            try:
                result = future.result()
                
                if result is None:
                    print(f"[{completed_count + 1}/{total_documents}] File {doc_id} processing completed with None result (status already logged internally).")
                    progress_tracker.finish_document_processing(worker_id, success=False, skipped=True)
                else:
                    print(f"[{completed_count + 1}/{total_documents}] Successfully processed: {result}")
                    log_processing_result(project_id, doc_id, "success")
                    progress_tracker.finish_document_processing(worker_id, success=True, pages=estimated_pages, size_mb=size_mb)
                    
            except Exception as e:
                print(f"[{completed_count + 1}/{total_documents}] Failed to process {doc_id}: {e}")
                log_processing_result(project_id, doc_id, "failure")
                progress_tracker.finish_document_processing(worker_id, success=False, skipped=False)
                
            completed_count += 1
            documents_processed += 1
            
            # Check time limit after each document completes
            if timed_mode and time_limit_seconds and start_time:
                elapsed_time = datetime.now() - start_time
                elapsed_minutes = elapsed_time.total_seconds() / 60
                remaining_seconds = max(0, time_limit_seconds - elapsed_time.total_seconds())
                remaining_minutes = remaining_seconds / 60
                
                if elapsed_time.total_seconds() >= time_limit_seconds:
                    print(f"[TIMED MODE] Time limit of {time_limit_seconds/60:.1f} minutes reached after processing {completed_count} documents.")
                    print(f"[TIMED MODE] Elapsed: {elapsed_minutes:.1f}min. Stopping gracefully.")
                    print(f"[TIMED MODE] Cancelling remaining {total_documents - completed_count} documents in queue.")
                    time_limit_reached = True
                    
                    # Cancel any remaining futures that haven't started yet
                    for remaining_future in future_to_task:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    
                    break
                
                # Show time status every 10 documents
                if completed_count % 10 == 0:
                    print(f"[TIMED MODE] Elapsed: {elapsed_minutes:.1f}min, Remaining: {remaining_minutes:.1f}min ({completed_count}/{total_documents} processed)")
            
            # DYNAMIC QUEUING: Only submit next document if time limit not reached
            if not time_limit_reached and document_index < total_documents:
                # Final time check before submitting new work
                if timed_mode and time_limit_seconds and start_time:
                    elapsed_time = datetime.now() - start_time
                    if elapsed_time.total_seconds() >= time_limit_seconds:
                        print(f"[TIMED MODE] Time limit reached. Not submitting additional documents.")
                        time_limit_reached = True
                        break
                
                next_task = document_tasks[document_index]
                next_file_key = file_keys[document_index]
                next_base_meta = metadata_list[document_index]
                next_api_doc = api_docs_list[document_index]
                
                next_doc_id = next_base_meta.get("document_id") or os.path.basename(next_file_key)
                next_doc_name = next_api_doc.get('name', next_doc_id)
                
                # File size handling for next document
                next_file_size_raw = next_api_doc.get('internalSize', '0')
                try:
                    next_file_size_bytes = int(next_file_size_raw) if next_file_size_raw else 0
                except (ValueError, TypeError):
                    next_file_size_bytes = 0
                
                next_size_mb = next_file_size_bytes / (1024 * 1024) if next_file_size_bytes > 0 else None
                next_estimated_pages = max(1, int(next_file_size_bytes / 50000)) if next_file_size_bytes > 0 else None
                
                # Submit next document immediately 
                next_future = executor.submit(load_data, next_file_key, next_base_meta, temp_dir, next_api_doc, is_retry)
                future_to_task[next_future] = (next_task, next_file_key, next_base_meta, next_api_doc, worker_id, next_doc_name, next_estimated_pages, next_size_mb, document_index)
                
                # Update progress tracker for new document
                progress_tracker.update_current_project(next_task.project_name)
                progress_tracker.start_document_processing(worker_id, next_doc_name, next_estimated_pages, next_size_mb)
                
                print(f"[DYNAMIC] Worker {worker_id} immediately started next document: {next_doc_name} ({document_index + 1}/{total_documents} queued)")
                document_index += 1
            elif document_index >= total_documents:
                print(f"[DYNAMIC] Worker {worker_id} finished. No more documents to assign. ({total_documents - completed_count} workers still active)")

    if time_limit_reached:
        print(f"[TIMED MODE] Graceful shutdown completed. Processed {documents_processed} documents before time limit.")
    else:
        print(f"All {total_documents} documents processed with dynamic queuing.")
    
    return {"time_limit_reached": time_limit_reached, "documents_processed": documents_processed}


def process_projects_in_parallel(projects, embedder_temp_dir, start_time, timed_mode, time_limit_seconds, retry_failed_only=False, retry_skipped_only=False, repair_mode=False, shallow_mode=False, shallow_limit=None):
    """Process multiple projects in parallel using a unified document queue across all projects"""
    from collections import namedtuple
    
    # Create a unified queue of all documents across all projects
    DocumentTask = namedtuple('DocumentTask', ['project_id', 'project_name', 's3_key', 'metadata', 'api_doc'])
    document_queue = []
    project_results = {}
    
    mode_desc = "new documents"
    if retry_failed_only and retry_skipped_only:
        mode_desc = "failed and skipped documents"
    elif retry_failed_only:
        mode_desc = "failed documents"
    elif retry_skipped_only:
        mode_desc = "skipped documents"
    elif repair_mode:
        mode_desc = "documents needing repair"
    
    print(f"\n=== Building unified document queue across {len(projects)} projects ({mode_desc}) ===")
    
    # Build the unified document queue
    for project in projects:
        project_id = project["_id"]
        project_name = project["name"]
        project_results[project_id] = {"project_name": project_name, "duration_seconds": 0}
        
        # Ensure project record exists
        upsert_project(project_id, project_name, project)
        
        print(f"Queuing documents from project: {project_name} ({project_id})")
        
        files_count = get_files_count_for_project(project_id)
        page_size = settings.api_pagination_settings.documents_page_size
        file_total_pages = (files_count + page_size - 1) // page_size
        
        already_completed = load_completed_files(project_id)
        already_incomplete = load_incomplete_files(project_id)
        already_skipped = load_skipped_files(project_id)
        
        documents_queued = 0
        
        # For shallow mode, track how many successful documents this project already has
        if shallow_mode:
            successful_count = len(already_completed)
            if successful_count >= shallow_limit:
                print(f"  Skipping {project_name} - already has {successful_count} successful documents (limit: {shallow_limit})")
                continue
            remaining_quota = shallow_limit - successful_count
            print(f"  {project_name} has {successful_count}/{shallow_limit} successful documents. Will queue up to {remaining_quota} more.")
        
        # Handle repair mode differently - get repair candidates
        if repair_mode:
            repair_candidates = get_repair_candidates_for_processing(project_id)
            
            if not repair_candidates:
                print(f"  No documents need repair for {project_name}")
                continue
            
            print(f"  Found {len(repair_candidates)} documents that need repair for {project_name}")
            
            # Process repair candidates
            for candidate in repair_candidates:
                doc_id = candidate['document_id']
                
                # Clean up existing inconsistent data first
                cleanup_summary = cleanup_document_data(doc_id, project_id)
                print(f"  Cleaned up inconsistent data for {candidate['document_name'][:50]}: {cleanup_summary['chunks_deleted']} chunks, {cleanup_summary['document_records_deleted']} docs, {cleanup_summary['processing_logs_deleted']} logs")
                
                # Find the document in the API to reprocess it
                doc_found = False
                for search_page in range(file_total_pages):
                    search_files = get_files_for_project(project_id, search_page, page_size)
                    for doc in search_files:
                        if doc["_id"] == doc_id:
                            doc_found = True
                            s3_key = doc.get("internalURL")
                            if s3_key:
                                doc_meta = format_metadata(project, doc)
                                document_task = DocumentTask(
                                    project_id=project_id,
                                    project_name=project_name,
                                    s3_key=s3_key,
                                    metadata=doc_meta,
                                    api_doc=doc
                                )
                                document_queue.append(document_task)
                                documents_queued += 1
                            break
                    if doc_found:
                        break
                
                if not doc_found:
                    print(f"  Warning: Could not find document {doc_id} in API - may have been deleted")
        else:
            # Normal mode: iterate through all files
            for file_page_number in range(file_total_pages):
                files_data = get_files_for_project(project_id, file_page_number, page_size)
                
                if not files_data:
                    continue
                    
                for doc in files_data:
                    doc_id = doc["_id"]
                    doc_name = doc.get('name', doc_id)
                    is_processed, status = is_document_already_processed(
                        doc_id, already_completed, already_incomplete, already_skipped
                    )
                    
                    # Handle retry modes
                    if retry_failed_only and retry_skipped_only:
                        # Combined mode: process both failed and skipped documents
                        if not is_processed or (status != "failed" and status != "skipped"):
                            continue
                        
                        status_type = "failed" if status == "failed" else "skipped"
                        print(f"  Queuing {status_type} document for retry: {doc_name}")
                        
                    elif retry_failed_only:
                        # Only process files that previously failed
                        if not is_processed or status != "failed":
                            continue
                        print(f"  Queuing failed document for retry: {doc_name}")
                        
                    elif retry_skipped_only:
                        # Only process files that were previously skipped
                        if not is_processed or status != "skipped":
                            continue
                        print(f"  Queuing skipped document for retry: {doc_name}")
                        
                    else:
                        # Normal mode: skip already processed files
                        if is_processed:
                            continue
                    
                    # Shallow mode: check if we've reached the quota for this project
                    if shallow_mode and documents_queued >= remaining_quota:
                        print(f"  Reached shallow limit for {project_name} ({documents_queued}/{remaining_quota} documents queued)")
                        break
                        
                    s3_key = doc.get("internalURL")
                    if not s3_key:
                        continue
                        
                    doc_meta = format_metadata(project, doc)
                    document_task = DocumentTask(
                        project_id=project_id,
                        project_name=project_name,
                        s3_key=s3_key,
                        metadata=doc_meta,
                        api_doc=doc
                    )
                    document_queue.append(document_task)
                    documents_queued += 1
        
        print(f"  Queued {documents_queued} documents from {project_name}")
    
    total_documents = len(document_queue)
    print(f"\n=== Unified queue built: {total_documents} documents across {len(projects)} projects ===")
    
    if total_documents == 0:
        print(f"No {mode_desc} to process across all projects")
        return list(project_results.values())
    
    # Process documents using true continuous queue (no batching)
    files_concurrency_size = settings.multi_processing_settings.files_concurrency_size
    
    print(f"Processing {total_documents} documents with {files_concurrency_size} workers using continuous queue")
    print(f"Workers will continuously pull from queue - no batching, no idle time")
    
    # Use the mixed-project processor with all documents at once
    # This creates a true continuous queue where workers pull documents as they finish
    processing_result = process_mixed_project_files(
        document_queue,  # Pass all documents as one big queue
        [task.s3_key for task in document_queue],
        [task.metadata for task in document_queue],
        [task.api_doc for task in document_queue],
        batch_size=files_concurrency_size,
        temp_dir=embedder_temp_dir,
        is_retry=(retry_failed_only or retry_skipped_only),
        timed_mode=timed_mode,
        time_limit_seconds=time_limit_seconds,
        start_time=start_time,
    )
    
    # Handle timed mode results
    if timed_mode and processing_result.get("time_limit_reached"):
        print(f"\n[TIMED MODE] Processing stopped due to time limit.")
        print(f"[TIMED MODE] Documents processed: {processing_result.get('documents_processed', 0)}")
        
        # Update progress tracking with actual processed count
        for _ in range(processing_result.get('documents_processed', 0)):
            progress_tracker.increment_processed()
    
    # Calculate final results
    for project_id in project_results:
        # For cross-project mode, we don't track individual project durations accurately
        # since documents are interleaved. Set a placeholder value.
        project_results[project_id]["duration_seconds"] = 0
    
    return list(project_results.values())


if __name__ == "__main__":   
    # Suppress the process pool error messages that occur during shutdown
    suppress_process_pool_errors()
    
    # Initialize OCR processor
    initialize_ocr()
    
    try:
        parser = argparse.ArgumentParser(
            description="Process projects and their documents."
        )
        parser.add_argument(
            "--project_id", type=str, nargs='+', help="The ID(s) of the project(s) to process. Can specify multiple: --project_id id1 id2 id3"
        )
        parser.add_argument(
            "--shallow", "-s", type=int, metavar="LIMIT", help="Enable shallow mode: process up to LIMIT successful documents per project and then move to the next project. Example: --shallow 5"
        )
        parser.add_argument(
            "--skip-hnsw-indexes", action="store_true", help="Skip creation of HNSW vector indexes for semantic search (faster startup, less resource usage)."
        )
        parser.add_argument(
            "--retry-failed", action="store_true", help="Only process documents that previously failed processing. Useful for retrying files that had errors."
        )
        parser.add_argument(
            "--retry-skipped", action="store_true", help="Only process documents that were previously skipped. Useful for retrying files that were skipped due to missing OCR or unsupported formats."
        )
        parser.add_argument(
            "--repair", action="store_true", help="Repair mode: identify and fix documents in inconsistent states (partial processing, orphaned chunks, failed but with data). Cleans up and reprocesses automatically."
        )
        parser.add_argument(
            "--reset", action="store_true", help="Reset mode: completely clean and reprocess a specific project. Deletes all documents, chunks, and processing logs for the project. Can only be used with --project_id (single project only)."
        )
        parser.add_argument(
            "--timed", type=int, metavar="MINUTES", help="Run in timed mode for the specified number of minutes, then gracefully stop. Example: --timed 60"
        )
        args = parser.parse_args()

        # Custom check for missing shallow limit value
        import sys
        if any(arg in sys.argv for arg in ["--shallow", "-s"]) and args.shallow is None:
            parser.error("Argument --shallow/-s requires an integer value. Example: --shallow 5")

        # Validate flag combinations
        # Note: --retry-failed and --retry-skipped can now be used together
        # to process both failed and skipped documents
        
        if args.repair and (args.retry_failed or args.retry_skipped):
            parser.error("Cannot use --repair with --retry-failed or --retry-skipped. Repair mode automatically handles inconsistent states.")
        
        # Validate reset flag - must be used only with a single project_id
        if args.reset:
            if not args.project_id:
                parser.error("--reset requires --project_id to be specified. Reset mode can only be used with a specific project.")
            if len(args.project_id) > 1:
                parser.error("--reset can only be used with a single project ID. Multiple projects are not allowed with reset mode.")
            if args.retry_failed or args.retry_skipped or args.repair or args.shallow:
                parser.error("--reset cannot be used with other processing modes (--retry-failed, --retry-skipped, --repair, --shallow). Reset mode can only be combined with --timed.")
        
        # Validate timed argument
        if args.timed is not None and args.timed <= 0:
            parser.error("Timed mode requires a positive number of minutes. Example: --timed 60")
        
        if args.retry_failed and args.shallow:
            print("WARNING: Using --retry-failed with --shallow mode. Shallow limit will apply to failed files being retried.")
        
        if args.retry_skipped and args.shallow:
            print("WARNING: Using --retry-skipped with --shallow mode. Shallow limit will apply to skipped files being retried.")
        
        if args.timed and args.shallow:
            print("WARNING: Using --timed with --shallow mode. Both time limit and shallow limit will apply.")

        shallow_mode = args.shallow is not None
        shallow_limit = args.shallow if shallow_mode else None
        timed_mode = args.timed is not None
        time_limit_minutes = args.timed if timed_mode else None

        if args.project_id:
            # Handle reset mode first (clean slate)
            if args.reset:
                project_id = args.project_id[0]  # We validated it's only one project
                print(f"[RESET] Starting complete reset for project {project_id}")
                
                # Perform complete cleanup
                cleanup_summary = cleanup_project_data(project_id)
                print(f"[RESET] Cleanup complete: {cleanup_summary}")
                
                # Now process normally (fresh start)
                print(f"[RESET] Beginning fresh processing for project {project_id}")
                result = process_projects([project_id], shallow_mode=False, shallow_limit=None, skip_hnsw_indexes=args.skip_hnsw_indexes, retry_failed_only=False, retry_skipped_only=False, repair_mode=False, timed_mode=timed_mode, time_limit_minutes=time_limit_minutes)
                print(f"[RESET] Fresh processing complete: {result}")
            else:
                # Run immediately if project_id(s) are provided (normal modes)
                result = process_projects(args.project_id, shallow_mode=shallow_mode, shallow_limit=shallow_limit, skip_hnsw_indexes=args.skip_hnsw_indexes, retry_failed_only=args.retry_failed, retry_skipped_only=args.retry_skipped, repair_mode=args.repair, timed_mode=timed_mode, time_limit_minutes=time_limit_minutes)
                print(result)
        else:
            # Run for all projects if no project_id is provided
            print("No project_id provided. Processing all projects.")
            result = process_projects(shallow_mode=shallow_mode, shallow_limit=shallow_limit, skip_hnsw_indexes=args.skip_hnsw_indexes, retry_failed_only=args.retry_failed, retry_skipped_only=args.retry_skipped, repair_mode=args.repair, timed_mode=timed_mode, time_limit_minutes=time_limit_minutes)
            print(result)
    finally:
        pass
