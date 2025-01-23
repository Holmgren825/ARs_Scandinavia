#!/usr/bin/bash

# Error handling function
error_handler() {
    echo "ERROR: Script failed at step: $1"
    exit 1
}

echo "Starting analysis pipeline..."

mamba activate ar_ident || error_handler "Environment activation" && \
echo "✓ Environment activated successfully" && \
python ./scripts/process_artmip.py || error_handler "Processing ARTMIP data" && \
echo "✓ ARTMIP processing complete" && \
python ./scripts/track_ars_artmip.py || error_handler "Tracking ARs" && \
echo "✓ AR tracking complete" && \
python ./scripts/collapse_tracked_ars.py || error_handler "Collapsing tracked ARs" && \
echo "✓ AR collapse complete" && \
python ./scripts/run_artmip_collapse_kmeans.py || error_handler "Running k-means analysis" && \
echo "✓ K-means analysis complete"

echo "Analysis pipeline completed successfully!"
