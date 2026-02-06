#!/bin/bash

# RAG System Setup Script
# This script sets up and tests the complete RAG pipeline

echo "=========================================="
echo "RAG SYSTEM SETUP"
echo "=========================================="

# Step 1: Check dependencies
echo ""
echo "1. Checking dependencies..."
python -c "import chromadb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ chromadb not found. Installing..."
    pip install chromadb==0.4.22
else
    echo "   ✅ chromadb installed"
fi

python -c "import sentence_transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ sentence-transformers not found. Installing..."
    pip install sentence-transformers==2.3.1
else
    echo "   ✅ sentence-transformers installed"
fi

# Step 2: Initialize RAG (index schemas)
echo ""
echo "2. Initializing RAG system (indexing schemas)..."
python init_rag.py
if [ $? -ne 0 ]; then
    echo "   ❌ RAG initialization failed"
    exit 1
fi

# Step 3: Test RAG pipeline
echo ""
echo "3. Testing RAG pipeline..."
python test_rag.py
if [ $? -ne 0 ]; then
    echo "   ❌ RAG testing failed"
    exit 1
fi

# Step 4: Success
echo ""
echo "=========================================="
echo "✅ RAG SYSTEM READY"
echo "=========================================="
echo ""
echo "The RAG system is now active and will:"
echo "  • Reduce prompt size from 27k to 5-10k chars"
echo "  • Retrieve only relevant schemas per query"
echo "  • Support 50+ datasets easily"
echo "  • Improve response speed and reduce costs"
echo ""
echo "RAG is enabled by default in ConversationManager."
echo "To disable: ConversationManager(use_rag=False)"
echo "=========================================="
