"""
Monitoring and Logging Module with Azure Application Insights.

This module provides:
- Azure Application Insights integration
- Structured logging
- Performance tracking decorators
- Error tracking
"""

import logging
import time
import functools
from typing import Callable, Any, Dict
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
from applicationinsights import TelemetryClient

from config import get_config

# Global telemetry client
_telemetry_client = None


def get_telemetry_client() -> TelemetryClient:
    """
    Get the global Application Insights telemetry client.

    Returns:
        TelemetryClient instance
    """
    global _telemetry_client

    if _telemetry_client is None:
        config = get_config()
        _telemetry_client = TelemetryClient(config.app_insights_connection_string)

    return _telemetry_client


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with Azure Application Insights integration.

    Args:
        log_level: Logging level (default: INFO)

    Returns:
        Configured root logger
    """
    config = get_config()

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler (for local debugging)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Azure Application Insights handler
    try:
        azure_handler = AzureLogHandler(
            connection_string=config.app_insights_connection_string
        )
        azure_handler.setLevel(log_level)
        logger.addHandler(azure_handler)
        logger.info("Azure Application Insights logging enabled")
    except Exception as e:
        logger.warning(f"Failed to enable Azure Application Insights: {e}")

    return logger


def track_performance(operation_name: str = None):
    """
    Decorator to track function performance.

    Logs execution time and sends metrics to Application Insights.

    Args:
        operation_name: Name of the operation (defaults to function name)

    Example:
        @track_performance("document_extraction")
        def extract_document(pdf_path):
            # ... extraction logic ...
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = operation_name or func.__name__
            logger = logging.getLogger(func.__module__)

            # Start tracking
            start_time = time.time()
            logger.info(f"Starting operation: {func_name}")

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Calculate duration
                duration = time.time() - start_time

                # Log success
                logger.info(
                    f"Operation '{func_name}' completed successfully in {duration:.2f}s"
                )

                # Track in Application Insights
                try:
                    client = get_telemetry_client()
                    client.track_metric(
                        name=f"{func_name}_duration",
                        value=duration,
                        properties={"status": "success"}
                    )
                    client.flush()
                except:
                    pass  # Don't fail if telemetry fails

                return result

            except Exception as e:
                # Calculate duration
                duration = time.time() - start_time

                # Log error
                logger.error(
                    f"Operation '{func_name}' failed after {duration:.2f}s: {e}"
                )

                # Track error in Application Insights
                try:
                    client = get_telemetry_client()
                    client.track_exception()
                    client.track_metric(
                        name=f"{func_name}_duration",
                        value=duration,
                        properties={"status": "error"}
                    )
                    client.flush()
                except:
                    pass  # Don't fail if telemetry fails

                # Re-raise the exception
                raise

        return wrapper
    return decorator


def track_event(event_name: str, properties: Dict[str, Any] = None):
    """
    Track a custom event in Application Insights.

    Args:
        event_name: Name of the event
        properties: Optional dictionary of event properties

    Example:
        track_event("document_uploaded", {"file_name": "doc.pdf", "size": 1024})
    """
    try:
        client = get_telemetry_client()
        client.track_event(event_name, properties or {})
        client.flush()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to track event '{event_name}': {e}")


def track_metric(metric_name: str, value: float, properties: Dict[str, Any] = None):
    """
    Track a custom metric in Application Insights.

    Args:
        metric_name: Name of the metric
        value: Metric value
        properties: Optional dictionary of metric properties

    Example:
        track_metric("chunks_indexed", 150, {"source": "doc.pdf"})
    """
    try:
        client = get_telemetry_client()
        client.track_metric(metric_name, value, properties=properties or {})
        client.flush()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to track metric '{metric_name}': {e}")


def log_operation(operation_name: str, properties: Dict[str, Any] = None):
    """
    Context manager for tracking operations.

    Args:
        operation_name: Name of the operation
        properties: Optional properties to track

    Example:
        with log_operation("indexing", {"doc_count": 10}):
            # ... indexing logic ...
            pass
    """
    class OperationLogger:
        def __init__(self, name: str, props: Dict[str, Any] = None):
            self.name = name
            self.properties = props or {}
            self.start_time = None
            self.logger = logging.getLogger(__name__)

        def __enter__(self):
            self.start_time = time.time()
            self.logger.info(f"Starting operation: {self.name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time

            if exc_type is None:
                # Success
                self.logger.info(
                    f"Operation '{self.name}' completed in {duration:.2f}s"
                )
                status = "success"
            else:
                # Error
                self.logger.error(
                    f"Operation '{self.name}' failed after {duration:.2f}s: {exc_val}"
                )
                status = "error"

            # Track in Application Insights
            try:
                self.properties.update({
                    "duration": duration,
                    "status": status
                })
                track_event(f"operation_{self.name}", self.properties)
            except:
                pass

            return False  # Don't suppress exceptions

    return OperationLogger(operation_name, properties)


class StructuredLogger:
    """
    Structured logger with Application Insights integration.

    Provides consistent logging with structured data.
    """

    def __init__(self, name: str):
        """
        Initialize structured logger.

        Args:
            name: Logger name (usually __name__)
        """
        self.logger = logging.getLogger(name)
        self.telemetry_client = get_telemetry_client()

    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, extra=kwargs)

        if kwargs:
            track_event(f"info_{message[:50]}", kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, extra=kwargs)

        if kwargs:
            track_event(f"warning_{message[:50]}", kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, extra=kwargs)

        if kwargs:
            track_event(f"error_{message[:50]}", kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, extra=kwargs)


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        StructuredLogger instance

    Example:
        logger = get_structured_logger(__name__)
        logger.info("Document processed", doc_name="file.pdf", chunks=10)
    """
    return StructuredLogger(name)


if __name__ == "__main__":
    # Test logging setup
    import sys

    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing Azure Application Insights integration")

    # Test performance tracking
    @track_performance("test_operation")
    def test_function():
        time.sleep(1)
        return "success"

    result = test_function()
    logger.info(f"Test function result: {result}")

    # Test event tracking
    track_event("test_event", {"key": "value", "count": 42})

    # Test metric tracking
    track_metric("test_metric", 123.45, {"category": "test"})

    # Test operation logging
    with log_operation("test_context_operation", {"test": True}):
        time.sleep(0.5)
        logger.info("Inside operation context")

    # Test structured logger
    struct_logger = get_structured_logger(__name__)
    struct_logger.info("Structured log test", module="monitoring", level="info")

    logger.info("All tests completed")
    print("Logging test completed. Check Azure Application Insights for telemetry data.")
